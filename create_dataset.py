from absl import flags, app
import pickle as pk
import numpy as np
from pyscf import gto, dft
from einsum import dm2rho, dm2rho01, dm2eT2, dm2eT, dm2eJ, dm2eK, cal_dipole, cal_I
from opt_einsum import contract

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_list('pkls',
                    default = ['/root/data/data/pkl_folder_CH_4/c2h6_0.0000.pkl',
                               '/root/data/data/pkl_folder_CH_4/c2h6_-0.1000.pkl',
                               '/root/data/data/pkl_folder_CH_4/c2h6_0.1000.pkl',],
                    help = 'a list of pickles')
  flags.DEFINE_boolean('val', default = False, help = 'whether to generate validation dataset')

def gen_cube(center, mol, dm, mesh, ni, ggnorm, *args):
    phi012 = ni.eval_ao(mol, np.array([center]), deriv=2)
    grad = dm2rho01(dm, phi012[:4])[1:, 0]
    phi2 = np.zeros((3,3)+phi012[0].shape)
    phi2[0,:] = phi012[4:7]
    phi2[1,0] = phi012[5]
    phi2[1,1] = phi012[7]
    phi2[1,2] = phi012[8]
    phi2[2,0] = phi012[6]
    phi2[2,1] = phi012[8]
    phi2[2,2] = phi012[9]
    I = 2*(contract('ij,pri,qrj->rpq',dm, phi012[1:4], phi012[1:4])+
            contract('ij,ri,pqrj->rpq',dm, phi012[0], phi2))[0]
    
    sort_index = np.argsort(-np.abs(grad))
    e, v = np.linalg.eigh(I)
    dot = np.dot(v[:,sort_index[0]], grad)
    if dot < 0:
        v[:,sort_index[0]] = -v[:,sort_index[0]]
    dot = np.dot(v[:,sort_index[1]], grad)
    if dot < 0:
        v[:,sort_index[1]] = -v[:,sort_index[1]]
    dot = np.dot(v[:,sort_index[2]], grad)
    if dot < 0:
        v[:,sort_index[2]] = -v[:,sort_index[2]]
    cube = center + v.dot(mesh.T).T
    phi01 = ni.eval_ao(mol, cube, deriv=1)
    assert len(dm) != 2 or np.allclose(dm[1], 0)
    rho01 = dm2rho01(dm, phi01)
    rho = rho01[0]
    gnorm = contract('dr,dr->r', rho01[1:], rho01[1:])**0.5
    tau = 0.5*contract('ij, dri, drj->r', dm, phi01[1:], phi01[1:])
    return np.hstack((rho.reshape(-1, 1), gnorm.reshape(-1, 1), tau.reshape(-1, 1)))

def gen_cube_old(center, mol, dm, mesh, ni, theta):    
    cube = center + mesh
    phi = ni.eval_ao(mol, cube, deriv=0)
    rho = dm2rho(dm, phi) # 1331
    R = cal_direction(cube, rho, theta)
    cube = center + R.dot(mesh.T).T
    phi01 = ni.eval_ao(mol, cube, deriv=1)
    assert len(dm) != 2 or np.allclose(dm[1], 0)
    rho01 = dm2rho01(dm, phi01)
    rho = rho01[0]
    gnorm = contract('dr,dr->r', rho01[1:], rho01[1:])**0.5
    tau = 0.5*contract('ij, dri, drj->r', dm, phi01[1:], phi01[1:])
    return np.hstack((rho.reshape(-1, 1), gnorm.reshape(-1, 1), tau.reshape(-1, 1)))

def gen_cube_parallel(centers, mol, dm, mesh, ni, theta):
    # print('start', time.ctime()) 
    cube_batch = centers + mesh[:,None,:] #1331, batch, 3
    # print(time.ctime())
    phi = ni.eval_ao(mol, cube_batch.reshape(-1, 3), deriv=0)
    # print(time.ctime())
    rho_batch = dm2rho(dm, phi).reshape(-1, len(centers)) # 1331
    new_cube = []
    # print(time.ctime())
    for rho, cube, center in zip(rho_batch.T, np.moveaxis(cube_batch, 0, 1), centers):
        R = cal_direction(cube, rho, theta)
        new_cube.append(center + R.dot(mesh.T).T)
    # print(time.ctime())
    phi = ni.eval_ao(mol, np.array(new_cube).reshape(-1,3), deriv=0)
    # print(time.ctime())
    assert len(dm) != 2 or np.allclose(dm[1], 0)
    rho = dm2rho(dm, phi).reshape(len(centers), -1)
    # print('end', time.ctime())
    return rho

def get_mesh(a, n_samples=6):
    pos = np.linspace(-a/2+a/(n_samples-1)*.5, a/2-a/(n_samples-1)*.5, n_samples-1)
    mesh = np.meshgrid(pos,pos,pos, indexing='ij')
    mesh = np.stack(mesh, axis=-1).reshape(-1,3)
    in_sphere_id = contract('ri,ri->r', mesh, mesh)<=(.5*a)**2
    mesh = mesh[in_sphere_id]
    return mesh

def get_feature(data, mol, ni, a, n_samples=6, ggnorm=False, dm='b3'):
    theta = (1.35*a/(n_samples-1))**2
    # N = 30
    mesh = get_mesh(a, n_samples)
    res = []
    for center in data['gc']:
        cube_b3 = gen_cube(center, mol, data['dm_'+dm], mesh, ni, ggnorm, theta)
        res.append(cube_b3)
    return np.array(res)

def cal_mu(gc, atoms_charges, atoms_coords):
    res = 0
    for z, c in zip(atoms_charges, atoms_coords):
        dc = np.linalg.norm(gc-c, axis=-1)
        res -= z/dc
    return res

def gen_train_data(path_list, info=['rho', 'gnorm', 'tau'], eps=1e-7, a=0.9, n_samples=6, dm='b3'):
    x, e, v, rho, e2, e3, rho_b3, gw = [], [], [], [], [], [], [], []
    for fn in path_list:
        print(fn)
        with open(fn, 'rb') as f:
            data = pk.load(f)
        mol = gto.M(atom=[str(n)+' '+' '.join(c.astype(str)) for n, c in
                zip(data['atoms_charges'], data['atoms_coords'])], basis=data['basis'],
                        spin=data['spin'], charge=data['charge'], unit='bohr', verbose=0)
        ni = dft.numint.NumInt()
        phi012 = ni.eval_ao(mol, data['gc'], deriv=2)
        data['phi01'] = phi012[:4]
        phi2 = phi012[[4,7,9],:]
        e_T_b3 = .5*dm2eT2(data['dm_b3'], data['phi01'][0], phi2)
        nu = mol.intor('int1e_grids_sph', grids=data['gc'])
        mu = cal_mu(data['gc'], data['atoms_charges'], data['atoms_coords'])
        e_ext_b3 = mu*data['rho_b3']
        e_ext_ccsd = mu*data['rho_ccsd']
        e_J_b3 = 0.5*dm2eJ(data['dm_b3'], data['phi01'][0], nu)
        e_J_ccsd = 0.5*dm2eJ(data['dm_ccsd'], data['phi01'][0], nu)
        e_xc_b3 = ni.eval_xc('b3lypg', data['rho01_b3'], data['spin'], relativity=0, deriv=0, verbose=None)[0]*data['rho_b3'] - 0.2*.25*dm2eK(data['dm_b3'], data['phi01'][0], nu)
        e_corr = data['e_T_ccsd']-e_T_b3+e_ext_ccsd-e_ext_b3+e_J_ccsd-e_J_b3+data['e_xc_noT']-e_xc_b3
        x.append(get_feature(data, mol, ni, a, n_samples, dm=dm))
        rho.append(data['rho_ccsd'])
        rho_b3.append(data['rho_b3'])
        e.append(e_corr/(data['rho_b3']+eps))
        gw.append(data['gw'])
    return np.vstack(x), np.concatenate(e), np.concatenate(rho), np.concatenate(rho_b3), path_list, np.concatenate(gw)

def gen_valid_data(path_list, info=['rho', 'gnorm', 'tau'], eps=1e-7, a=0.9, n_samples=6, dm='b3'):
    x, e, v, rho, rho_b3, gw = [], [], [], [], [], []
    for fn in path_list:
        print(fn)
        with open(fn, 'rb') as f:
            data = pk.load(f)
        mol = gto.M(atom=[str(n)+' '+' '.join(c.astype(str)) for n, c in
                zip(data['atoms_charges'], data['atoms_coords'])], basis=data['basis'],
                        spin=data['spin'], charge=data['charge'], unit='bohr', verbose=0)
        ni = dft.numint.NumInt()
        phi012 = ni.eval_ao(mol, data['gc'], deriv=2)
        data['phi01'] = phi012[:4]
        phi2 = phi012[[4,7,9],:]
        e_T_b3 = .5*dm2eT2(data['dm_b3'], data['phi01'][0], phi2)
        nu = mol.intor('int1e_grids_sph', grids=data['gc'])
        mu = cal_mu(data['gc'], data['atoms_charges'], data['atoms_coords'])
        e_ext_b3 = mu*data['rho_b3']
        e_ext_ccsd = mu*data['rho_ccsd']
        e_J_b3 = 0.5*dm2eJ(data['dm_b3'], data['phi01'][0], nu)
        e_J_ccsd = 0.5*dm2eJ(data['dm_ccsd'], data['phi01'][0], nu)
        e_xc_b3 = ni.eval_xc('b3lypg', data['rho01_b3'], data['spin'], relativity=0, deriv=0, verbose=None)[0]*data['rho_b3'] - 0.2*.25*dm2eK(data['dm_b3'], data['phi01'][0], nu)
        e_corr = data['e_T_ccsd']-e_T_b3+e_ext_ccsd-e_ext_b3+e_J_ccsd-e_J_b3+data['e_xc_noT']-e_xc_b3
        x.append(get_feature(data, mol, ni, a, n_samples, dm=dm))
        rho.append(data['rho_ccsd'])
        rho_b3.append(data['rho_b3'])
        e.append(e_corr/(data['rho_b3']+eps))
        gw.append(data['gw'])
    return np.vstack(x), np.concatenate(e), np.concatenate(rho), np.concatenate(rho_b3), path_list, np.concatenate(gw)

def main(unused_argv):
  if not FLAGS.val:
    x, e, _, _, _, _ = gen_train_data(FLAGS.pkls, a = 0.9, n_samples = 6)
    np.savez('trainset.npy', datasets = x, labels = e)
  else:
    x, e, _, _, _, _ = gen_valid_data(FLAGS.pkls, a = 0.9, n_samples = 6)
    np.savez('valset.npy', datasets = x, labels = e)

from torch.utils.data import Dataset

class RhoDataset(Dataset):
  def __init__(self, datasets):
    data = np.load(datasets)
    self.datasets = np.copy(data['datasets'])
    self.labels = np.copy(data['labels'])
    del data
    assert self.datasets.shape[0] == self.labels.shape[0]
  def __len__(self):
    return self.datasets.shape[0]
  def __getitem__(self, index):
    dataset, label = self.datasets[index], self.labels[index]
    # NOTE: dataset.shape = (81, 3), label.shape = ()
    dataset = np.reshape(dataset, (81 * 3,))
    return np.log(dataset.astype(np.float32)), np.arcsinh(label.astype(np.float32))

if __name__=="__main__":
    add_options()
    app.run(main)

