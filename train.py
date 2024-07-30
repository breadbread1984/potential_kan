#!/usr/bin/python3

from absl import flags, app
from os import mkdir
from os.path import exists, join
import numpy as np
import torch
from torch import device, save, load, no_grad, any, isnan, autograd, sinh, log
from torch.nn import L1Loss
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, distributed
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from create_dataset import RhoDataset
from models import PredictorSmall

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('trainset', default = None, help = 'file for trainset')
  flags.DEFINE_string('valset', default = None, help = 'file for valset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to directory for checkpoints')
  flags.DEFINE_integer('batch_size', default = 1000, help = 'batch size')
  flags.DEFINE_integer('save_freq', default = 1000, help = 'checkpoint save frequency')
  flags.DEFINE_integer('epochs', default = 200, help = 'epochs to train')
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')
  flags.DEFINE_integer('decay_steps', default = 200000, help = 'decay steps')
  flags.DEFINE_integer('workers', default = 16, help = 'number of workers')
  flags.DEFINE_float('reg_weight', default = 0.01, help = 'weight of regularizer')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = ['cpu', 'cuda'], help = 'device')

def main(unused_argv):
  autograd.set_detect_anomaly(True)
  trainset = RhoDataset(FLAGS.trainset)
  evalset = RhoDataset(FLAGS.valset)
  dist.init_process_group(backend='nccl')
  torch.cuda.set_device(dist.get_rank())
  trainset_sampler = distributed.DistributedSampler(trainset)
  evalset_sampler = distributed.DistributedSampler(evalset)
  if dist.get_rank() == 0:
    print('trainset size: %d, evalset size: %d' % (len(trainset), len(evalset)))
  train_dataloader = DataLoader(trainset, batch_size = FLAGS.batch_size, shuffle = False, num_workers = FLAGS.workers, sampler = trainset_sampler, pin_memory = False)
  eval_dataloader = DataLoader(evalset, batch_size = FLAGS.batch_size, shuffle = False, num_workers = FLAGS.workers, sampler = evalset_sampler, pin_memory = False)
  model = PredictorSmall(in_channel = 1)
  model.to(device(FLAGS.device))
  model = DDP(model, device_ids=[dist.get_rank()], output_device=dist.get_rank(), find_unused_parameters=True)
  mae = L1Loss()
  optimizer = Adam(model.parameters(), lr = FLAGS.lr)
  scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 5, T_mult = 2)
  if dist.get_rank() == 0:
    if not exists(FLAGS.ckpt): mkdir(FLAGS.ckpt)
    tb_writer = SummaryWriter(log_dir = join(FLAGS.ckpt, 'summaries'))
  start_epoch = 0
  if exists(join(FLAGS.ckpt, 'model.pth')):
    ckpt = load(join(FLAGS.ckpt, 'model.pth'))
    state_dict = {(key.replace('module.','') if key.startswith('module.') else key):value for key, value in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = ckpt['scheduler']
    start_epoch = ckpt['epoch']
  for epoch in range(start_epoch, FLAGS.epochs):
    train_dataloader.sampler.set_epoch(epoch)
    model.train()
    for step, (x, e) in enumerate(train_dataloader):
      optimizer.zero_grad()
      x, e = x.to(device(FLAGS.device)), e.to(device(FLAGS.device))
      preds = model(x)
      loss = mae(e, preds)
      loss.backward()
      optimizer.step()
      global_steps = epoch * len(train_dataloader) + step
      if global_steps % 100 == 0 and dist.get_rank() == 0:
        print('Step #%d Epoch #%d: loss %f, lr %f' % (global_steps, epoch, loss, scheduler.get_last_lr()[0]))
        tb_writer.add_scalar('loss', loss, global_steps)
    scheduler.step()
    if dist.get_rank() == 0:
      eval_dataloader.sampler.set_epoch(epoch)
      model.eval()
      true, diff = list(), list()
      for x, e in eval_dataloader:
        x, e = x.to(device(FLAGS.device)), e.to(device(FLAGS.device))
        pred = model(x)
        true_e = torch.sinh(e).detach().cpu().numpy()
        pred_e = torch.sinh(pred).detach().cpu().numpy()
        true.append(true_e)
        diff.append(np.abs(true_e - pred_e))
      true = np.squeeze(np.concatenate(true, axis = 0))
      diff = np.squeeze(np.concatenate(diff, axis = 0))
      plt.xlabel('exc ground truth')
      plt.ylabel('exc prediction absolute loss')
      plt.scatter(true, diff, c = 'b', s = 2, alpha = 0.7)
      global_steps = epoch * len(train_dataloader) + step
      tb_writer.add_figure('loss distribution', plt.gcf(), global_steps)
      plt.clf()
      counts, bins = np.histogram(diff, bins = 50)
      plt.stairs(counts, bins)
      tb_writer.add_figure('histogram', plt.gcf(), global_steps)
    if dist.get_rank() == 0:
      ckpt = {'epoch': epoch,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler}
      save(ckpt, join(FLAGS.ckpt, 'model-ep%d.pth' % epoch))

if __name__ == "__main__":
  add_options()
  app.run(main)

