#!/usr/bin/python3

from absl import flags, app
from os.path import exists, join
import numpy as np
import torch
from torch import device, save, load, no_grad, any, isnan, autograd, sinh, log
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from create_dataset import RhoDataset
from models import KAN

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('valset', default = None, help = 'file for valset')
  flags.DEFINE_integer('batch_size', default = 4096, help = 'batch size')
  flags.DEFINE_string('ckpt', default = None, help = 'checkpoint')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')
  flags.DEFINE_integer('workers', default = 256, help = 'number of workers')

def main(unused_argv):
  assert exists(FLAGS.ckpt)
  evalset = RhoDataset(FLAGS.valset)
  eval_dataloader = DataLoader(evalset, batch_size = FLAGS.batch_size, shuffle = False, num_workers = FLAGS.workers)
  model = KAN(channels = [81*3, 8, 4, 1], grid = 7, k = 3)
  ckpt = load(FLAGS.ckpt)
  state_dict = {(key.replace('module.','') if key.startswith('module.') else key):value for key, value in ckpt['state_dict'].items()}
  model.load_state_dict(state_dict)
  model.to(device(FLAGS.device))
  model.eval()
  true = list()
  diff = list()
  for x, e in eval_dataloader:
    x, e = x.to(device(FLAGS.device)), e.to(device(FLAGS.device))
    pred, _ = model(x, do_train = False)
    true_e = torch.sinh(e).detach().cpu().numpy()
    pred_e = torch.sinh(pred).detach().cpu().numpy()
    true.append(true_e)
    diff.append(np.abs(true_e - pred_e))
  true = np.squeeze(np.concatenate(true, axis = 0))
  diff = np.squeeze(np.concatenate(diff, axis = 0))
  plt.xlabel('exc ground truth')
  plt.ylabel('exc prediction absolute loss')
  plt.scatter(exc_trues, diffs, c = 'b', s = 2, alpha = 0.7)
  plt.savefig('loss_plot.png')
  plt.clf()
  counts, bins = np.histogram(diff, bins = 100)
  plt.stairs(counts, bins)
  plt.savefig('histogram.png')
  print('mean absolute error: %f median absolute error: %f' % (np.mean(diff), np.median(diff)))

if __name__ == "__main__":
  add_options()
  app.run(main)

