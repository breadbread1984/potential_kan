#!/usr/bin/python3

from os import walk
from os.path import join, splitext
import numpy as np
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
        self.start_indices.append(count)
        self.data_count += np.prod(self.npys[-1]['rho_inv_4_norm'].shape[1:])
  def __len__(self):
    return self.data_count
  def __getitem__(self, index):
    memmap_index = bisect(self.start_indices, index) - 1
    index_in_memmap = index - self.start_indices[memmap_index]
    data = self.npys[memmap_index]['rho_inv_4_norm'].reshape((4,-1))[index_in_memmap] # data.shape = (4,)
    vxc = self.npys[memmap_index]['vxc1_b3lyp'].reshape((1,-1))[index_in_memmap] # vxc.shape = (1,)
    exc = self.npys[memmap_index]['exc1_tr_b3lyp'].reshape((1,-1))[index_in_memmap] # exc.shape = (1,)
    return np.log(dataset.astype(np.float32)), np.arcsinh(label.astype(np.float32))

if __name__=="__main__":
  pass
