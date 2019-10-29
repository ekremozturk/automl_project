import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from darts_model_search import Network
from darts_architect import Architect

import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
from datetime import datetime

from datasets import K49
from datasets import KMNIST



parser = argparse.ArgumentParser("darts")
parser.add_argument('--d_data', type=str, default='KMNIST', help='location of the data corpus')
parser.add_argument('--d_batch_size', type=int, default=64, help='batch size')
parser.add_argument('--d_learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--d_learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--d_momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--d_weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--d_report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--d_gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--d_epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--d_init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--d_layers', type=int, default=8, help='total number of layers')
parser.add_argument('--d_model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--d_cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--d_cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--d_drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--d_save', type=str, default='EXP', help='experiment name')
parser.add_argument('--d_seed', type=int, default=2, help='random seed')
parser.add_argument('--d_grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--d_train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--d_unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--d_arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--d_arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args, unknowns = parser.parse_known_args()

CIFAR_CLASSES = 10

def main(exp_dir = None):

  ### LOGGING ###

  if(exp_dir == None):
    exp_dir = 'experiment-{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S%f"))
    utils.create_exp_dir(exp_dir)

  #args.d_save = exp_dir+'/darts_search'
  #utils.create_exp_dir(args.d_save, scripts_to_save=glob.glob('*.py'))

  #log_format = '%(asctime)s %(message)s'
  #logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
  #fh = logging.FileHandler(os.path.join(args.d_save, 'log.txt'))
  #fh.setFormatter(logging.Formatter(log_format))
  #logging.getLogger().addHandler(fh)

  ###############

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  #np.random.seed(args.d_seed)
  torch.cuda.set_device(args.d_gpu)
  cudnn.benchmark = True
  #torch.manual_seed(args.d_seed)
  cudnn.enabled=True
  #torch.cuda.manual_seed(args.d_seed)
  logging.info('gpu device = %d' % args.d_gpu)
  logging.info("args = %s", args)

  ########

  data_dir = '../data'
  data_augmentations = None

  if data_augmentations is None:
    # You can add any preprocessing/data augmentation you want here
    data_augmentations = transforms.ToTensor()
  elif isinstance(type(data_augmentations), list):
    data_augmentations = transforms.Compose(data_augmentations)
  elif not isinstance(data_augmentations, transforms.Compose):
    raise NotImplementedError

  train_data = None
  if(args.d_data == 'K49'):
    train_data = K49(data_dir, True, data_augmentations)
    CIFAR_CLASSES = 49
  else:
    train_data = KMNIST(data_dir, True, data_augmentations)

  #########

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.d_init_channels, CIFAR_CLASSES, args.d_layers, criterion)
  model = model.cuda()
  logging.info("Param size = %fMB", utils.count_parameters_in_MB(model))
  logging.info('Total # of params: %d', sum(p.numel() for p in model.parameters()))

  optimizer = torch.optim.SGD(model.parameters(), args.d_learning_rate, momentum=args.d_momentum, weight_decay=args.d_weight_decay)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.d_train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.d_batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.d_batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.d_epochs), eta_min=args.d_learning_rate_min)

  architect = Architect(model, args)
  for epoch in range(args.d_epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    if(epoch == args.d_epochs-1):
      architecture_res = exp_dir+'/arch'
      with open(architecture_res, 'wb') as f:
        pickle.dump(genotype, f)

    logging.info(F.softmax(model.alphas_normal, dim=-1))
    logging.info(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    #utils.save(model, os.path.join(args.d_save, 'weights.pt'))

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.d_unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.d_grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.d_report_freq == 0:
      logging.info('train step: %03d loss: %e accuracy: %f top5 accuracy: %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input).cuda()
      target = Variable(target).cuda(non_blocking=True)

      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.d_report_freq == 0:
        logging.info('valid step: %03d loss: %e accuracy: %f top5 accuracy: %f', step, objs.avg, top1.avg, top5.avg)
        
  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

