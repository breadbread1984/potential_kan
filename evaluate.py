#!/usr/bin/python3

from absl import flags, app
from os.path import exists, join
import numpy as np
import torch
from torch import device, save, load, no_grad, any, isnan, autograd, sinh, log
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from create_dataset import RhoDataset
from models import PredictorSmall

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('valset', default = None, help = 'directory for valset')
  flags.DEFINE_integer('batch_size', default = 4096, help = 'batch size')
  flags.DEFINE_string('ckpt', default = None, help = 'checkpoint')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')
  flags.DEFINE_integer('workers', default = 256, help = 'number of workers')

def main(unused_argv):
  assert exists(FLAGS.ckpt)
  evalset = RhoDataset(FLAGS.valset)
  eval_dataloader = DataLoader(evalset, batch_size = FLAGS.batch_size, shuffle = False, num_workers = FLAGS.workers)
  model = PredictorSmall(in_channel = 1)
  ckpt = load(FLAGS.ckpt)
  state_dict = {(key.replace('module.','') if key.startswith('module.') else key):value for key, value in ckpt['state_dict'].items()}
  model.load_state_dict(state_dict)
  model.to(device(FLAGS.device))
  model.eval()
  e_true, e_diff = list(), list()
  v_true, v_diff = list(), list()
  for rho, vxc, exc in eval_dataloader:
    rho, vxc, exc = rho.to(device(FLAGS.device)), vxc.to(device(FLAGS.device)), exc.to(device(FLAGS.device))
    rho.requires_grad = True
    inputs = torch.unsqueeze(rho, dim = -1)
    pred_exc = model(inputs)
    pred_vxc = autograd.grad(torch.sum(rho * pred_exc), rho, create_graph = True)[0]
    pred_exc = pred_exc.flatten()
    pred_vxc = pred_vxc.flatten()
    vxc = vxc.flatten()
    exc = exc.flatten()
    true_e = exc.detach().cpu().numpy()
    pred_e = pred_exc.detach().cpu().numpy()
    e_true.append(true_e)
    e_diff.append(np.abs(true_e - pred_e))
    true_v = vxc.detach().cpu().numpy()
    pred_v = pred_vxc.detach().cpu().numpy()
    v_true.append(true_v)
    v_diff.append(np.abs(true_v - pred_v))
  e_true = np.squeeze(np.concatenate(e_true, axis = 0))
  e_diff = np.squeeze(np.concatenate(e_diff, axis = 0))
  v_true = np.squeeze(np.concatenate(v_true, axis = 0))
  v_diff = np.squeeze(np.concatenate(v_diff, axis = 0))
  plt.xlabel('exc ground truth')
  plt.ylabel('exc prediction absolute loss')
  plt.scatter(e_true, e_diff, c = 'b', s = 2, alpha = 0.7)
  global_steps = epoch * len(train_dataloader) + step
  plt.savefig('exc_loss_distribution.png')
  plt.clf()
  plt.xlabel('vxc ground truth')
  plt.ylabel('vxc prediction absolute loss')
  plt.scatter(v_true, v_diff, c = 'b', s = 2, alpha = 0.7)
  plt.savefig('vxc_loss_distribution.png')
  plt.clf()
  counts, bins = np.histogram(e_diff, bins = 50)
  plt.stairs(counts, bins)
  plt.savefig('exc_histogram.png')
  plt.clf()
  counts, bins = np.histogram(v_diff, bins = 50)
  plt.stairs(counts, bins)
  plt.savefig('vxc histogram')
  print('1) exc:')
  print('mean absolute error: %f' % np.mean(e_diff))
  print('median absolute error: %f' % np.median(e_diff))
  print('90%% quantile: %f' % np.quantile(e_diff, 0.9))
  print('95%% quantile: %f' % np.quantile(e_diff, 0.95))
  print('99%% quantile: %f' % np.quantile(e_diff, 0.99))
  print('2) vxc:')
  print('mean absolute error: %f' % np.mean(v_diff))
  print('median absolute error: %f' % np.median(v_diff))
  print('90%% quantile: %f' % np.quantile(v_diff, 0.9))
  print('95%% quantile: %f' % np.quantile(v_diff, 0.95))
  print('99%% quantile: %f' % np.quantile(v_diff, 0.99))

if __name__ == "__main__":
  add_options()
  app.run(main)

