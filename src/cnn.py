import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from utils import AvgrageMeter, accuracy, drop_path, plot_confusion_matrix
from operations import *

import sys
import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

'''
Ref: https://arxiv.org/abs/1806.09055
'''
class Cell(nn.Module):

  def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    logging.info('Number of feature maps | Prev prev: %d | Prev: %d | Current: %d' % (C_prev_prev, C_prev, C))

    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
    
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)

'''
Ref: https://arxiv.org/abs/1806.09055
'''
class dartsModel(nn.Module):

    def __init__(self, genotype, config, input_shape=(1, 28, 28), num_classes = 10):
        super(dartsModel, self).__init__()

        self.report_freq = 50
        self.drop_path_prob = -1

        layers = config['n_cells']
        C = config['init_channels']
        stem_multiplier = 3

        C_curr = int(stem_multiplier*C)
        self.stem = nn.Sequential(nn.ZeroPad2d((1,0,1,0)), nn.Conv2d(input_shape[0], C_curr, 2, padding=0, bias=False), nn.ReLU(inplace=False))

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [1, 3]: # 1: first normal, 0:first reduction
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr

        self.fc_layers = nn.ModuleList()
        n_in = self._get_conv_output(input_shape)
        n_out = config['n_hidden_units']
        for i in range(config['n_hidden_layers']):
            fc = nn.Linear(int(n_in), int(n_out))
            self.fc_layers += [fc]
            n_in = n_out
            n_out /= 2

        self.classifier = nn.Linear(int(n_in), num_classes)
        self.dropout = nn.Dropout(p=config['dropout_ratio'])

    def _get_conv_output(self, shape):
        bs = 1
        with torch.no_grad():
            s1 = s0 = self.stem(Variable(torch.rand(bs, *shape)))
            for i, cell in enumerate(self.cells):
                s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            n_size = s1.data.view(bs, -1).size(1)
        return n_size

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
        out = s1.view(s1.size(0), -1)
        for fc_layer in self.fc_layers:
            out = self.dropout(F.relu(fc_layer(out)))
        logits = self.classifier(out.view(out.size(0),-1)) # Here it flattens inside
        return logits

    def train_fn(self, optimizer, criterion, loader, device, train=True):
        score = AvgrageMeter()
        objs = AvgrageMeter()
        self.train()

        for step, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = self(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            acc, _ = accuracy(logits, labels, topk=(1, 5))
            n = images.size(0)
            objs.update(loss.item(), n)
            score.update(acc.item(), n)

            if step % self.report_freq == 0:
                logging.info('Training | step: %d | loss: %e | accuracy: %f' % (step, objs.avg, score.avg))

        return score.avg, objs.avg

    def eval_fn(self, loader, device, train=False, confusion_m = False, criterion = None):
        """
        Evaluation method
        :param loader: data loader for either training or testing set
        :param device: torch device
        :param train: boolean to indicate if training or test set is used
        :return: accuracy on the data
        """
        objs = AvgrageMeter()
        score = AvgrageMeter()
        self.eval()
        with torch.no_grad():
            for step, (images, labels) in enumerate(loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = self(images)

                acc, _ = accuracy(outputs, labels, topk=(1, 5))
                score.update(acc.item(), images.size(0))

                if(criterion):
                    loss = criterion(outputs, labels)
                    objs.update(loss.item(), images.size(0))

                if step % self.report_freq == 0:
                    logging.info('Evaluation | step: %d | accuracy: %f' % (step, score.avg))   

        return score.avg, objs.avg

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class torchModel(nn.Module):
    def __init__(self, config, input_shape=(1, 28, 28), num_classes=10):
        super(torchModel, self).__init__()
        layers = []
        n_layers = config['n_hidden_layers']
        n_conv_layers = 1
        kernel_size = 2
        in_channels = input_shape[0]
        out_channels = 4

        for i in range(n_conv_layers):
            c = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=2, padding=1
                         )
            a = nn.ReLU(inplace=False)
            p = nn.MaxPool2d(kernel_size=2, stride=1)
            layers.extend([c, a, p])
            in_channels = out_channels
            out_channels *= 2

        self.conv_layers = nn.Sequential(*layers)
        self.output_size = num_classes

        self.fc_layers = nn.ModuleList()
        n_in = self._get_conv_output(input_shape)
        n_out = 256
        for i in range(n_layers):
            fc = nn.Linear(int(n_in), int(n_out))
            self.fc_layers += [fc]
            n_in = n_out
            n_out /= 2

        self.last_fc = nn.Linear(int(n_in), self.output_size)
        self.dropout = nn.Dropout(p=0.2)

    # generate input sample and forward to get shape
    def _get_conv_output(self, shape):
        bs = 1
        input = Variable(torch.rand(bs, *shape))
        output_feat = self.conv_layers(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        for fc_layer in self.fc_layers:
            x = self.dropout(F.relu(fc_layer(x)))
        x = self.last_fc(x)
        return x

    def train_fn(self, optimizer, criterion, loader, device, train=True):
        """
        Training method
        :param optimizer: optimization algorithm
        :criterion: loss function
        :param loader: data loader for either training or testing set
        :param device: torch device
        :param train: boolean to indicate if training or test set is used
        :return: (accuracy, loss) on the data
        """
        score = AvgrageMeter()
        objs = AvgrageMeter()
        self.train()

        t = tqdm(loader)
        for images, labels in t:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = self(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            acc, _ = accuracy(logits, labels, topk=(1, 5))
            n = images.size(0)
            objs.update(loss.item(), n)
            score.update(acc.item(), n)

            t.set_description('(=> Training) Loss: {:.4f}'.format(objs.avg))

        return score.avg, objs.avg

    def eval_fn(self, loader, device, train=False, confusion_m = False, criterion = None):
        """
        Evaluation method
        :param loader: data loader for either training or testing set
        :param device: torch device
        :param train: boolean to indicate if training or test set is used
        :return: accuracy on the data
        """
        objs = AvgrageMeter()
        score = AvgrageMeter()
        self.eval()

        t = tqdm(loader)
        with torch.no_grad():
            for images, labels in t:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self(images)
                acc, _ = accuracy(outputs, labels, topk=(1, 5))
                score.update(acc.item(), images.size(0))

                if(criterion):
                  loss = criterion(outputs, labels)
                  objs.update(loss.data, images.size(0))

                if(confusion_m):
                  # Plot confusion matrix
                  plot_confusion_matrix(labels.cpu(), outputs.topk(1, 1, True, True)[1].cpu(), normalize = True, title='Confusion matrix')

                t.set_description('(=> Test) Score: {:.4f}'.format(score.avg))

        return score.avg, objs.avg

