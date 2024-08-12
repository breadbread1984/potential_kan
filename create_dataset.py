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
        self.npys.append(np.load(join(root, f), mmap_mode = 'r'))
        self.start_indices.append(self.data_count)
        self.data_count += self.npys[-1]['rho_inv_4_norm'].shape[1]
  def __len__(self):
    return self.data_count
  def __getitem__(self, index):
    memmap_index = bisect(self.start_indices, index) - 1
    index_in_memmap = index - self.start_indices[memmap_index]
    rho = self.npys[memmap_index]['rho_inv_4_norm'][0,index_in_memmap] # data.shape = (75, 302,)
    vxc = self.npys[memmap_index]['vxc1_b3lyp'][index_in_memmap] # vxc.shape = (75, 302,)
    exc = self.npys[memmap_index]['exc1_tr_b3lyp'][index_in_memmap] # exc.shape = (75, 302,)
    rho = np.ascontiguousarray(rho)
    vxc = np.ascontiguousarray(vxc)
    exc = np.ascontiguousarray(exc)
    return rho.astype(np.float32), vxc.astype(np.float32), exc.astype(np.float32)

if __name__=="__main__":
  pass
