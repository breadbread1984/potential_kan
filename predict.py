#!/usr/bin/python3

from os.path import join
import numpy as np
import torch
from torch import load, device, autograd
from models import PredictorSmall

class Predict(object):
  def __init__(self, ckpt_path, precision = 'float32'):
    assert precision in {'float32', 'float64'}
    ckpt = load(join(ckpt_path, 'model.pth'))
    self.model = PredictorSmall(in_channel = 1).to(torch.device('cuda'))
    state_dict = {(key.replace('module.','') if key.startswith('module.') else key):value for key, value in ckpt['state_dict'].items()}
    self.model.load_state_dict(state_dict)
    self.model.eval()
    self.precision = precision
  def get_v(self, scf_r_3, grids):
    rho = grids.vector_to_matrix(scf_r_3[0,:]) # rho.shape = (natom, 75, 302)
    rho = torch.tensor(rho, dtype = self.precision).to(device('cuda'))
    inputs = torch.unsqueeze(rho, dim = -1) # inputs.shape = (natom, 75, 302, 1)
    inputs.requires_grad = True
    pred_exc = self.model(inputs) # pred_exc.shape = (natom, 75, 302)
    pred_vxc = autograd.grad(torch.sum(rho * pred_exc), rho, create_graph = True)[0] + pred_exc # pred_exc.shape = (natom, 75, 302)
    pred_vxc = pred_vxc.detach().cpu().numpy()
    vxc_scf = grids.matrix_to_vector(pred_vxc)
    return vxc_scf
  def get_e(self, scf_r_3, grids):
    rho = grids.vector_to_matrix(scf_r_3[0,:]) # rho.shape = (natom, 75, 302)
    rho = torch.tensor(rho, dtype = self.precision).to(device('cuda'))
    inputs = torch.unsqueeze(rho, dim = -1) # inputs.shape = (natom, 75, 302, 1)
    inputs.requires_grad = True
    pred_exc = self.model(inputs) # pred_exc.shape = (natom, 75, 302)
    exc_scf = grids.matrix_to_vector(pred_exc)
    return exc_scf

