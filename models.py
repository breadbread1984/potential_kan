#!/usr/bin/python3

import math
import torch
from torch import nn

class Attention(nn.Module):
  def __init__(self, **kwargs):
    super(Attention, self).__init__()
    self.channel = kwargs.get('channel', 768)
    self.num_heads = kwargs.get('num_heads', 8)
    self.qkv_bias = kwargs.get('qkv_bias', False)
    self.drop_rate = kwargs.get('drop_rate', 0.1)

    self.dense1 = nn.Linear(self.channel, self.channel * 3, bias = self.qkv_bias)
    self.dense2 = nn.Linear(self.channel, self.channel, bias = self.qkv_bias)
    self.dropout1 = nn.Dropout(self.drop_rate)
    self.dropout2 = nn.Dropout(self.drop_rate)
  def forward(self, inputs):
    # inputs.shape = (batch, seq_len, channel)
    results = self.dense1(inputs) # results.shape = (batch, 302, 3 * channel)
    b, s, _ = results.shape
    results = torch.reshape(results, (b, s, 3, self.num_heads, self.channel // self.num_heads)) # results.shape = (batch, seq_len, 3, head, channel // head)
    results = torch.permute(results, (0, 2, 3, 1, 4)) # results.shape = (batch, 3, head, seq_len, channel // head)
    q, k, v = results[:,0,...], results[:,1,...], results[:,2,...] # shape = (batch, head, seq_len, channel // head)
    qk = torch.matmul(q, torch.transpose(k, 2, 3)) # qk.shape = (batch, head, seq_len, seq_len)
    attn = torch.softmax(qk, dim = -1) # attn.shape = (batch, head, seq_len, seq_len)
    attn = self.dropout1(attn)
    qkv = torch.permute(torch.matmul(attn, v), (0, 2, 1, 3)) # qkv.shape = (batch, seq_len, head, channel // head)
    qkv = torch.reshape(qkv, (b, s, self.channel)) # qkv.shape = (batch, seq_len, channel)
    results = self.dense2(qkv) # results.shape = (batch, seq_len, channel)
    results = self.dropout2(results)
    return results

class ABlock(nn.Module):
  def __init__(self, length, **kwargs):
    super(ABlock, self).__init__()
    self.channel = kwargs.get('channel', 768)
    self.mlp_ratio = kwargs.get('mlp_ratio', 4)
    self.drop_rate = kwargs.get('drop_rate', 0.1)
    self.num_heads = kwargs.get('num_heads', 8)
    self.qkv_bias = kwargs.get('qkv_bias', False)

    self.dense1 = nn.Linear(self.channel, self.channel * self.mlp_ratio)
    self.dense2 = nn.Linear(self.channel * self.mlp_ratio, self.channel)
    self.gelu = nn.GELU()
    self.layernorm1 = nn.LayerNorm([length, self.channel,])
    self.layernorm2 = nn.LayerNorm([length, self.channel,])
    self.dropout0 = nn.Dropout(self.drop_rate)
    self.dropout1 = nn.Dropout(self.drop_rate)
    self.dropout2 = nn.Dropout(self.drop_rate)
    self.atten = Attention(**kwargs)
  def forward(self, inputs):
    # inputs.shape = (batch, length, channel)
    # attention
    skip = inputs
    results = self.layernorm1(inputs) # results.shape = (batch, length, channel)
    results = self.atten(results) # results.shape = (batch, length, channel)
    results = self.dropout0(results)
    results = skip + results
    # mlp
    skip = results
    results = self.layernorm2(results)
    results = self.dense1(results) # results.shape = (batch, length, channel * mlp_ratio)
    results = self.gelu(results)
    results = self.dropout1(results)
    results = self.dense2(results) # results.shape = (batch, length, channel)
    results = self.dropout2(results)
    results = skip + results
    return results

class Extractor(nn.Module):
  def __init__(self, **kwargs):
    super(Extractor, self).__init__()
    self.in_channel = kwargs.get('in_channel', 1)
    self.hidden_channels = kwargs.get('hidden_channels', 512)
    self.depth = kwargs.get('depth', 12)
    self.mlp_ratio = kwargs.get('mlp_ratio', 4.)
    self.drop_rate = kwargs.get('drop_rate', 0.1)
    self.qkv_bias = kwargs.get('qkv_bias', False)
    self.num_heads = kwargs.get('num_heads', 8)
    
    self.gelu = nn.GELU()
    self.tanh = nn.Tanh()
    self.dense1 = nn.Linear(self.in_channel, self.hidden_channels)
    self.dense2 = nn.Linear(self.hidden_channels, self.hidden_channels)
    self.layernorm1 = nn.LayerNorm([75, 302, 1,])
    self.layernorm2 = nn.LayerNorm([75, 302, self.hidden_channels])
    self.dropout1 = nn.Dropout(self.drop_rate)
    self.layer_blocks = nn.ModuleList([ABlock(channel = self.hidden_channels, qkv_bias = self.qkv_bias, num_heads = self.num_heads, length = 75, **kwargs) for i in range(self.depth)])
    self.spatial_blocks = nn.ModuleList([ABlock(channel = self.hidden_channels, qkv_bias = self.qkv_bias, num_heads = self.num_heads, length = 302, **kwargs) for i in range(self.depth)])
    self.head = nn.Linear(self.hidden_channels, 1, bias = False)
  def forward(self, inputs):
    batch = inputs.shape[0]
    # inputs.shape = (batch, 75, 302, 1)
    results = self.layernorm1(inputs)
    results = self.dense1(results) # results.shape = (batch, 75, 302, hidden_channels)
    results = self.gelu(results)
    results = self.dropout1(results)
    # do attention only when the feature shape is small enough
    for i in range(self.depth):
      # results.shape = (batch, 75, 302, hidden_channels)
      results = torch.reshape(results, (batch * 75, 302, self.hidden_channels)) # results.shape = (batch * 75, 302, hidden_channels)
      results = self.spatial_blocks[i](results) # results.shape = (batch * 75, 302, hidden_channels)
      results = torch.reshape(results, (batch, 75, 302, self.hidden_channels)) # results.shape = (batch, 75, 302, hidden_channels)
      results = torch.permute(results, (0,2,1,3)) # result.shape = (batch, 302, 75, hidden_channels)
      results = torch.reshape(results, (batch * 302, 75, self.hidden_channels)) # results.shape = (batch * 302, 75, hidden_channels)
      results = self.layer_blocks[i](results) # results.shape = (batch * 302, 75, hidden_channels)
      results = torch.reshape(results, (batch, 302, 75, self.hidden_channels)) # results.shape = (batch, 302, 75, hidden_channels)
      results = torch.permute(results, (0,2,1,3)) # results.shape = (batch, 75, 302, hidden_channel)
    results = self.layernorm2(results)
    results = self.dense2(results) # results.shape = (batch, 75, 302, hidden_channels)
    results = self.tanh(results) # results.shape = (batch, 75, 302, hidden_channels)
    results = self.head(results) # results.shape = (batch, 75, 302, 1)
    return results

class PredictorSmall(nn.Module):
  def __init__(self, **kwargs):
    super(PredictorSmall, self).__init__()
    hidden_channels = kwargs.get('hidden_channels', 256)
    depth = kwargs.get('depth', 3)
    self.predictor = Extractor(hidden_channels = hidden_channels, depth = depth, **kwargs)
  def forward(self, inputs):
    return torch.squeeze(self.predictor(inputs), dim = -1)

class PredictorBase(nn.Module):
  def __init__(self, **kwargs):
    super(PredictorBase, self).__init__()
    hidden_channels = kwargs.get('hidden_channels', 512)
    depth = kwargs.get('depth', 24)
    self.predictor = Extractor(hidden_channels = hidden_channels, depth = depth, **kwargs)
  def forward(self, inputs):
    return torch.squeeze(self.predictor(inputs), dim = -1)

if __name__ == "__main__":
  predictor = PredictorSmall(in_channel = 1)
  inputs = torch.randn(2, 75, 302, 1)
  results = predictor(inputs)
  print(results.shape)
