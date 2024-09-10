#!/usr/bin/python3

import math
import torch
from torch import nn

class MLPMixer(nn.Module):
  def __init__(self, **kwargs):
    super(MLPMixer, self).__init__()
    self.hidden_dim = kwargs.get('hidden_dim', 768)
    self.num_blocks = kwargs.get('num_blocks', 12)
    self.tokens_mlp_dim = kwargs.get('tokens_mlp_dim', 384)
    self.channels_mlp_dim = kwargs.get('channels_mlp_dim', 3072)
    self.drop_rate = kwargs.get('drop_rate', 0.1)

    self.layernorm1 = nn.LayerNorm((75, 302, 1))
    self.dense = nn.Linear(1, self.hidden_dim)
    self.gelu = nn.GELU()
    self.dropout = nn.Dropout(self.drop_rate)
    layers = dict()
    for i in range(self.num_blocks):
      layers.update({
        'layernorm1_%d' % i: nn.LayerNorm((self.hidden_dim, 302)),
        'linear1_%d' % i: nn.Linear(302, self.tokens_mlp_dim),
        'gelu1_%d' % i: nn.GELU(),
        'linear2_%d' % i: nn.Linear(self.tokens_mlp_dim, 302),
        'layernorm2_%d' % i: nn.LayerNorm((302, self.hidden_dim)),
        'linear3_%d' % i: nn.Linear(self.hidden_dim, self.channels_mlp_dim),
        'gelu2_%d' % i: nn.GELU(),
        'linear4_%d' % i: nn.Linear(self.channels_mlp_dim, self.hidden_dim),

        'layernorm3_%d' % i: nn.LayerNorm((self.hidden_dim, 75)),
        'linear5_%d' % i: nn.Linear(75, self.tokens_mlp_dim),
        'gelu3_%d' % i: nn.GELU(),
        'linear6_%d' % i: nn.Linear(self.tokens_mlp_dim, 75),
        'layernorm4_%d' % i: nn.LayerNorm((75, self.hidden_dim)),
        'linear7_%d' % i: nn.Linear(self.hidden_dim, self.channels_mlp_dim),
        'gelu4_%d' % i: nn.GELU(),
        'linear8_%d' % i: nn.Linear(self.channels_mlp_dim, self.hidden_dim),
      })
    self.layers = nn.ModuleDict(layers)
    self.layernorm2 = nn.LayerNorm((75, 302, self.hidden_dim))
    self.head = nn.Linear(self.hidden_dim, 1)
  def forward(self, inputs):
    batch = inputs.shape[0]
    # inputs.shape = (batch, 75, 302, 1)
    results = self.layernorm1(inputs)
    results = self.dense(results) # results.shape = (batch, 75, 302, hidden_dim)
    results = self.gelu(results)
    results = self.dropout(results)

    for i in range(self.num_blocks):
      # merge dimension
      results = torch.reshape(results, (batch * 75, 302, self.hidden_dim))
      # 1) spatial mixing
      skip = results
      results = torch.permute(results, (0,2,1)) # results.shape = (batch, channel, 302)
      results = self.layers['layernorm1_%d' % i](results)
      results = self.layers['linear1_%d' % i](results) # results.shape = (batch, channel, token_mlp_dim)
      results = self.layers['gelu1_%d' % i](results)
      results = self.layers['linear2_%d' % i](results) # results.shape = (batch, channel, 302)
      results = torch.permute(results, (0,2,1)) # resutls.shape = (batch, 302, channel)
      results = results + skip
      # 2) channel mixing
      skip = results
      results = self.layers['layernorm2_%d' % i](results)
      results = self.layers['linear3_%d' % i](results) # results.shape = (batch, 302, channels_mlp_dim)
      results = self.layers['gelu2_%d' % i](results)
      results = self.layers['linear4_%d' % i](results) # results.shape = (batch, 302, channel)
      results = results + skip
      # remerge dimension
      results = torch.reshape(results, (batch, 75, 302, self.hidden_dim))
      results = torch.permute(results, (0,2,1,3))
      results = torch.reshape(results, (batch * 302, 75, self.hidden_dim)) # results.shape = (batch * 302, 75, channel)
      # 1) spatial mixing
      skip = results
      results = torch.permute(results, (0,2,1)) # results.shape = (batch, channel, 75)
      results = self.layers['layernorm3_%d' % i](results)
      results = self.layers['linear5_%d' % i](results) # results.shape = (batch, channel, token_mlp_dim)
      results = self.layers['gelu3_%d' % i](results)
      results = self.layers['linear6_%d' % i](results) # results.shape = (batch, channel, 75)
      results = torch.permute(results, (0,2,1)) # results.shape = (batch, 75, channel)
      results = results + skip
      # 2) channel mixing
      skip = results
      results = self.layers['layernorm4_%d' % i](results)
      results = self.layers['linear7_%d' % i](results) # results.shape = (batch, 75, channels_mlp_dim)
      results = self.layers['gelu4_%d' % i](results)
      results = self.layers['linear8_%d' % i](results) # results.shape = (batch, 75, channel)
      results = results + skip
      # reshape dimension
      results = torch.reshape(results, (batch, 302, 75, self.hidden_dim))
      results = torch.permute(results, (0,2,1,3)) # results.shape = (batch, 75, 302, channel)

    results = self.layernorm2(results) # results.shape = (batch, 75, 302, channel)
    results = self.head(results) # results.shape = (batch, 75, 302, 1)
    return results

class Predictor(nn.Module):
  def __init__(self, **kwargs):
    super(Predictor, self).__init__()
    self.predictor = MLPMixer(**kwargs)
  def forward(self, inputs):
    results = self.predictor(inputs)
    results = torch.squeeze(results, dim = -1) # results.shape = (batch, 75, 302)
    return results

class PredictorSmall(nn.Module):
  def __init__(self):
    super(PredictorSmall, self).__init__()
    kwargs = {'hidden_dim': 256, 'num_blocks': 12, 'tokens_mlp_dim': 384, 'channels_mlp_dim': 256*4, 'drop_rate': 0.1}
    self.predictor = Predictor(**kwargs)
  def forward(self, inputs):
    return self.predictor(inputs)

class PredictorBase(nn.Module):
  def __init__(self):
    super(PredictorBase, self).__init__()
    kwargs = {'hidden_dim': 768, 'num_blocks': 12, 'tokens_mlp_dim': 384, 'channels_mlp_dim': 3072, 'drop_rate': 0.1}
    self.predictor = Predictor(**kwargs)
  def forward(self, inputs):
    return self.predictor(inputs)

if __name__ == "__main__":
  predictor = PredictorSmall()
  inputs = torch.randn(2, 75, 302, 1)
  results = predictor(inputs)
  print(results.shape)
