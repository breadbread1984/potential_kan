#!/usr/bin/python3

import torch
from torch import nn

class MLP(nn.Module):
  def __init__(self,):
    super(MLP, self).__init__()
    self.model = nn.Sequential(
      nn.BatchNorm1d(81 * 3),
      nn.Linear(81 *3, 8),
      nn.Dropout(),
      nn.GELU(),
      nn.BatchNorm1d(8),
      nn.Linear(8,4),
      nn.Dropout(),
      nn.GELU(),
      nn.BatchNorm1d(4),
      nn.Linear(4,1)
    )
  def forward(self, x):
    return self.model(x)

if __name__ == "__main__":
  x = torch.randn(4,81*3)
  mlp = MLP()
  y = mlp(x)
  print(y.shape)
