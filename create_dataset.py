#!/usr/bin/python3

from os import walk
from os.path import join, splitext
import numpy as np
from bisect import bisect
import torch
from torch.utils.data import Dataset

class RhoDataset(Dataset):
  def __init__(self, dataset_dir):
    self.npys = list()
    self.start_indices = list()
    self.data_count = 0
    for root, dirs, files in walk(dataset_dir):
      for f in files:
        stem, ext = splitext(f)
        if ext != '.npz': continue
        fp = np.load(join(root, f))
        self.npys.append({'rho_inv_4_norm': fp['rho_inv_4_norm'],
                          'vxc1_lda': fp['vxc1_lda'],
                          'exc1_tr_lda': fp['exc1_tr_lda'],
                          'weights': fp['weights']})
        del fp
        self.start_indices.append(self.data_count)
        self.data_count += self.npys[-1]['rho_inv_4_norm'].shape[1]
  def __len__(self):
    return self.data_count
  def __getitem__(self, index):
    memmap_index = bisect(self.start_indices, index) - 1
    index_in_memmap = index - self.start_indices[memmap_index]
    rho = self.npys[memmap_index]['rho_inv_4_norm'][0,index_in_memmap] # data.shape = (75, 302,)
    vxc = self.npys[memmap_index]['vxc1_lda'][index_in_memmap] # vxc.shape = (75, 302,)
    exc = self.npys[memmap_index]['exc1_tr_lda'][index_in_memmap] # exc.shape = (75, 302,)
    weights = self.npys[memmap_index]['weights'][index_in_memmap]
    energy = np.sum(rho * exc * weights)
    rho = np.ascontiguousarray(rho)
    vxc = np.ascontiguousarray(vxc)
    exc = np.ascontiguousarray(exc)
    weights = np.ascontiguousarray(weights)
    return rho.astype(np.float32), vxc.astype(np.float32), exc.astype(np.float32), weights.astype(np.float32), energy.astype(np.float32)

if __name__=="__main__":
  pass
