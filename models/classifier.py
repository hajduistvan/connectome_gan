# -*- coding: utf-8 -*-
"""
@author István Hajdu at MTA TTK
https://github.com/hajduistvan/connectome_gan
"""
import torch.nn as nn
import torch
from ast import literal_eval as make_tuple
import os
from torchnet.meter import MovingAverageValueMeter
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np
from decimal import Decimal


class ConnectomeConvNet(nn.Module):

    def __init__(
            self,
            layers,
            lr,
            mom,
            wd,
            train_loader,
            val_loader,
            gpu_id,
            val_interval,
            run_id,
            log_dir,
            max_epochs,
            allow_stop=True,
            verbose=True
    ):
        super(ConnectomeConvNet, self).__init__()
        # self.config = config
        # self.class_architecture = self.config.ARCHITECTURE.CLASS
        self.exp_name = 'c1_' + str(layers[0]) + '_c2_' + str(layers[1]) + '_lr_' + '%.2E' % Decimal(
            lr) + '_mom_' + '%.2E' % Decimal(mom) + '_wd_' + '%.2E' % Decimal(wd)
        # self.val_loader = val_loader
        self.layers = layers
        self.log_dir = log_dir
        self.lr = lr
        self.mom = mom
        self.wd = wd
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gpu_id = gpu_id
        self.val_interval = val_interval
        self.run_id = run_id
        self.max_epochs = max_epochs
        self.allow_stop = allow_stop
        self.verbose = verbose
        self.layer1 = nn.Sequential(nn.Conv2d(1, self.layers[0], (1, 55)), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(self.layers[0], self.layers[1], (55, 1)), nn.ReLU())
        # self.layer23 = nn.Sequential(nn.Linear(self.layers[1], self.layers[1]), nn.ReLU())
        self.layer3 = nn.Linear(self.layers[1], 1)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.mom, weight_decay=self.wd,
                                         nesterov=True)
        self.cuda(self.gpu_id)
        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if not m.bias is None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if not m.bias is None:
                    torch.nn.init.constant_(m.bias, 0)
        # Loss
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_val_loss = np.inf
        self.best_val_acc = 0

        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.writer = SummaryWriter(self.log_dir)
        self.global_step = 0

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(-1, self.layers[1])
        x = self.layer3(x).view(-1)
        return x

    def train_step(self, inputs):
        data, labels = inputs
        data = data.cuda(self.gpu_id)
        labels = labels.view(-1).cuda(self.gpu_id)
        self.optimizer.zero_grad()

        outputs = self(data)
        self.loss = self.criterion(outputs, labels)
        self.loss.backward()
        if self.verbose:
            self.writer.add_scalars('train_loss', {self.exp_name: self.loss.detach()}, self.global_step)

        self.optimizer.step()
        self.global_step += 1

    def get_acc(self, outputs, labels):
        threshold = 0.5
        """
        Operates on detached cuda tensors!
        For binary classification
        :param outputs:
        :param labels:
        :return:
        """
        preds = (torch.sigmoid(outputs) >
                 threshold).long().squeeze(0)
        total = labels.size(0)
        correct = (preds == labels.long()).sum().item()
        accuracy = correct / total
        return accuracy

    def validate(self):
        self.eval()
        outputs, labels_list = [], []
        with torch.no_grad():
            for i, datapair in enumerate(self.val_loader):
                data, labels = datapair
                data, labels = data.cuda(self.gpu_id), labels.cuda(self.gpu_id)
                out = self(data)
                outputs.append(out)
                labels_list.append(labels)
            outputs = torch.cat(outputs)
            labels = torch.cat(labels_list).view(-1)
            val_loss = self.criterion(outputs, labels).cpu().numpy()
            val_acc = self.get_acc(outputs, labels)

            if val_loss < self.best_val_loss:
                self.best_val_acc = self.get_acc(outputs, labels)
                self.best_val_loss = val_loss

                save_dict = {
                    'state_dict': self.state_dict(),
                    'layers': self.layers,
                    'lr': self.lr,
                    'mom': self.mom,
                    'wd': self.wd,
                    'val_loss': val_loss,
                    'val_acc': self.best_val_acc
                }
                torch.save(save_dict, os.path.join(self.log_dir, str(self.gpu_id) +
                                                   str(self.run_id) + ".pth"), )

        self.train()
        return val_loss, val_acc

    def run_train(self):
        self.train()
        self.val_losses = []
        for epoch in tqdm(
                range(1, self.max_epochs),
                total=self.max_epochs,
                leave=False,
                dynamic_ncols=True,
        ):
            trainiter = iter(self.train_loader)
            for inputs in trainiter:
                self.train_step(inputs)
            val_loss, val_acc = self.validate()
            if self.verbose:
                self.writer.add_scalars('val_loss', {self.exp_name: val_loss}, self.global_step)
                self.writer.add_scalars('val_acc', {self.exp_name: val_acc}, self.global_step)
            self.val_losses.append(val_loss)
            if self.allow_stop and self.stop_iter():
                print("\nBreaking out at epoch ", epoch)
                break

        return self.best_val_loss, self.best_val_acc, self.num_parameters

    def stop_iter(self):
        """
        Watches val loss, and detects overfitting.
        crit1: True if all last 'buffer_length' losses are within 'treshold' * 'last_loss' range of
         their mean, meaning that the training froze/converged.
        crit2: True if last loss is greater than the other in 'buffer_length'.
        crit3: True if the last losses are more than 'multiplier' times greater than the best val loss.

        :return: should_stop: Bool, whether model has overfitted and training should stop
        """
        buffer_length = 20
        multiplier = 3
        treshold = 0.001
        if len(self.val_losses) <= buffer_length:
            return False

        crit1, crit2 = True, True
        for i in range(buffer_length - 1):
            crit2 = crit2 and self.val_losses[-(i + 2)] < self.val_losses[-1]
            crit1 = crit1 and np.abs(np.mean(self.val_losses[-buffer_length:]) - self.val_losses[-(i + 2)]) < treshold * \
                    self.val_losses[-(i + 2)]
        crit3 = np.median(self.val_losses[-buffer_length:]) > multiplier * self.best_val_loss
        should_stop = crit1 or crit3 and crit2
        return should_stop

    def test(self, test_loader):
        state_dict = torch.load(os.path.join(self.log_dir, str(self.gpu_id) + str(self.run_id) + ".pth"))['state_dict']
        self.load_state_dict(state_dict)
        self.eval()
        outputs, labels_list = [], []
        with torch.no_grad():
            for i, datapair in enumerate(test_loader):
                data, labels = datapair
                data, labels = data.cuda(self.gpu_id), labels.cuda(self.gpu_id)
                out = self(data)
                outputs.append(out)
                labels_list.append(labels)
            outputs = torch.cat(outputs)
            labels = torch.cat(labels_list).view(-1)
            test_loss = self.criterion(outputs, labels).cpu().numpy()
            test_acc = self.get_acc(outputs, labels)
        self.train()
        return test_loss, test_acc


class ConnectomeConvInferenceNet(nn.Module):
    def __init__(
            self,
            layers,
            state_dict,
            gpu_id,
            log_dir=None,
    ):
        super(ConnectomeConvInferenceNet, self).__init__()
        self.c1, self.c2 = layers
        self.gpu_id = gpu_id
        # self.log_dir = log_dir
        self.layer1 = nn.Sequential(nn.Conv2d(1, self.c1, (1, 55)), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(self.c1, self.c2, (55, 1)), nn.ReLU())
        self.layer3 = nn.Linear(self.c2, 1)
        self.cuda(self.gpu_id)
        self.load_state_dict(state_dict)

    def forward(self, x):
        self.eval()
        # with torch.no_grad:
        x1 = self.layer1(x)  # shape: [batch_size, c1, 55, 1]
        x1_no_nonlin = self.layer1[0](x)

        x2 = self.layer2(x1).view(-1, self.c2)  # shape: [batch_size, c2]
        x2_no_nonlin = self.layer2[0](x1)

        x3 = self.layer3(x2).view(-1)  # shape: [batch_size]
        return x1.view(-1, self.c1 * 55), x2_no_nonlin.view(-1, self.c2), x3.view(-1, 1)
