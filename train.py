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
from kan import KAN

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('trainset', default = None, help = 'file for trainset')
  flags.DEFINE_string('valset', default = None, help = 'file for valset')
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to directory for checkpoints')
  flags.DEFINE_integer('batch_size', default = 6, help = 'batch size')
  flags.DEFINE_integer('save_freq', default = 1000, help = 'checkpoint save frequency')
  flags.DEFINE_integer('epochs', default = 200, help = 'epochs to train')
  flags.DEFINE_float('lr', default = 1e-4, help = 'learning rate')
  flags.DEFINE_integer('decay_steps', default = 200000, help = 'decay steps')
  flags.DEFINE_integer('workers', default = 16, help = 'number of workers')
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
  model = KAN(width = [75 * 302, 8, 4, 75 * 302], grid = 7 , k = 3)
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
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = ckpt['scheduler']
    start_epoch = ckpt['epoch']
  for epoch in range(start_epoch, FLAGS.epochs):
    train_dataloader.sampler.set_epoch(epoch)
    model.train()
    for step, (rho, vxc, exc, weights, energy) in enumerate(train_dataloader):
      optimizer.zero_grad()
      # rho.shape = (batch, 75, 302, 1) vxc.shape = (batch, 75, 302) exc.shape = (batch, 75, 302)
      rho, vxc, exc, weights, energy = rho.to(device(FLAGS.device)), vxc.to(device(FLAGS.device)), exc.to(device(FLAGS.device)), weights.to(device(FLAGS.device)), energy.to(device(FLAGS.device))
      rho.requires_grad = True
      inputs = torch.flatten(rho, start_dim = 1) # inputs.shape = (batch, 75 * 302)
      if epoch == 0 and step % 5 == 0 and step < 50:
        model.module.update_grid(inputs)
      pred_exc = model(inputs)
      pred_exc = torch.reshape(pred_exc, (-1, 75, 302)) # pred_exc.shape = (batch, 75, 302)
      loss1 = mae(exc, pred_exc)
      
      pred_vxc = autograd.grad(torch.sum(rho * pred_exc), rho, create_graph = True)[0]
      loss2 = mae(vxc, pred_vxc)

      loss3 = mae(energy, torch.sum(pred_exc * rho * weights, dim = (1,2)))

      loss4 = model.get_reg('edge_forward_spline_n', 1., 2., 0., 0.)

      loss = loss1 + loss2 + loss3 + loss4
      loss.backward()
      optimizer.step()
      global_steps = epoch * len(train_dataloader) + step
      if global_steps % 100 == 0 and dist.get_rank() == 0:
        print('Step #%d Epoch #%d: exc loss %f, vxc loss %f, lr %f' % (global_steps, epoch, loss1, loss2, scheduler.get_last_lr()[0]))
        tb_writer.add_scalar('exc mae loss', loss1, global_steps)
        tb_writer.add_scalar('vxc mae loss', loss2, global_steps)
        tb_writer.add_scalar('e * rho * weights mae loss', loss3, global_steps)
    scheduler.step()
    if dist.get_rank() == 0:
      eval_dataloader.sampler.set_epoch(epoch)
      model.eval()
      e_true, e_diff = list(), list()
      v_true, v_diff = list(), list()
      for rho, vxc, exc, weights, energy in eval_dataloader:
        rho, vxc, exc = rho.to(device(FLAGS.device)), vxc.to(device(FLAGS.device)), exc.to(device(FLAGS.device))
        rho.requires_grad = True
        inputs = torch.flatten(rho, start_dim = 1) # inputs.shape = (batch, 75 * 302)
        pred_exc, _ = model(inputs, do_train = False) # pred_exc.shape = (batch, 75, 302)
        pred_exc = torch.reshape(pred_exc, (-1, 75, 302))
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
      tb_writer.add_figure('exc loss distribution', plt.gcf(), global_steps)
      plt.clf()
      plt.xlabel('vxc ground truth')
      plt.ylabel('vxc prediction absolute loss')
      plt.scatter(v_true, v_diff, c = 'b', s = 2, alpha = 0.7)
      global_steps = epoch * len(train_dataloader) + step
      tb_writer.add_figure('vxc loss distribution', plt.gcf(), global_steps)
      plt.clf()
      counts, bins = np.histogram(e_diff, bins = 50)
      plt.stairs(counts, bins)
      tb_writer.add_figure('exc histogram', plt.gcf(), global_steps)
      plt.clf()
      counts, bins = np.histogram(v_diff, bins = 50)
      plt.stairs(counts, bins)
      tb_writer.add_figure('vxc histogram', plt.gcf(), global_steps)
    if dist.get_rank() == 0:
      ckpt = {'epoch': epoch,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler}
      save(ckpt, join(FLAGS.ckpt, 'model-ep%d.pth' % epoch))

if __name__ == "__main__":
  add_options()
  app.run(main)
