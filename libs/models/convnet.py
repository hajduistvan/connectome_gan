# -*- coding: utf-8 -*-
"""
@author István Hajdu at MTA TTK
https://github.com/hajduistvan/connectome_gan
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
from ast import literal_eval as make_tuple
import os
from torchnet.meter import MovingAverageValueMeter
from tensorboardX import SummaryWriter
from tqdm import tqdm
import numpy as np


class ConnectomeConvNet(nn.Module):

    def __init__(
            self,
            config,
            mix,
            val_loader,
            gan_best_valid_loss=np.inf,
            rnd_init=False,
    ):
        super(ConnectomeConvNet, self).__init__()
        self.config = config
        self.class_architecture = self.config.ARCHITECTURE.CLASS
        self.mix = mix
        self.val_loader = val_loader
        self.set_dirs()
        self.h = self.config.MATR_SIZE
        self.rnd_init = rnd_init
        self.chs = make_tuple(self.class_architecture.CHS)
        self.neurons = make_tuple(self.class_architecture.NEURONS)
        self.p = self.class_architecture.DROPOUT
        self.best_valid_loss = np.inf
        self.gan_best_valid_loss = gan_best_valid_loss
        self.criterion, self.criterion_to_log = self.get_loss_fn()

        self.writer = SummaryWriter(self.log_dir)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.chs[0], (self.h, 1)),
            nn.BatchNorm2d(self.chs[0]),
            nn.ReLU(),
            nn.Dropout2d(self.p),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(self.chs[0], self.chs[1], (1, self.h)),
            nn.BatchNorm2d(self.chs[1]),
            nn.ReLU(),
            nn.Dropout2d(self.p),
        )
        self.block3 = nn.Sequential(
            nn.Linear(self.chs[1], self.neurons[0]),
            nn.BatchNorm1d(self.neurons[0]),
            nn.ReLU(),
            nn.Dropout(self.p),
        )
        self.block4 = nn.Sequential(
            nn.Linear(self.neurons[0], self.neurons[1]),
            nn.BatchNorm1d(self.neurons[1]),
            nn.ReLU(),
            nn.Dropout(self.p),
        )
        self.out = nn.Sequential(
            nn.Linear(self.neurons[1], self.neurons[2]),
        )
        self.optimizer = self.get_optimizer()
        self.initialize_weights()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(-1, self.chs[1])
        x = self.block3(x)
        x = self.block4(x)
        x = self.out(x)
        return x

    def train_step(self, inputs):
        data, labels = inputs
        data = data.cuda()
        labels = labels.view(-1, 1).cuda()
        self.optimizer.zero_grad()

        outputs = self(data)
        self.loss = self.criterion(outputs, labels)
        self.loss_log = self.criterion_to_log(outputs, labels).detach().cpu()
        self.loss.backward()
        self.optimizer.step()
        self.hook_fn()

    def hook_fn(self):
        self.train_loss_meter.add(self.loss_denorm_fn(self.loss_log))
        if not self.mix and self.global_step % self.config.ITER_TB.CLASS == 0:
            self.writer.add_scalar("classifier_training_loss", self.train_loss_meter.value()[0], self.global_step)
        if not self.mix and self.global_step % self.config.ITER_SAVE.CLASS == 0:
            torch.save(
                self.state_dict(),
                os.path.join(self.save_dir,
                             "checkpoint_class_{}.pth".format(self.global_step)),
            )
        if self.global_step % self.config.ITER_VAL.CLASS == 0:
            self.validate()

    def validate(self):
        self.eval()
        valid_loss_l = torch.cuda.FloatTensor(1).fill_(0)
        b = 0
        with torch.no_grad():
            for i, datapair in enumerate(self.val_loader):
                data, labels = datapair
                batch_len = data.shape[0]
                labels = labels.type(torch.FloatTensor).view(-1, 1)
                data, labels = data.cuda(), labels.cuda()
                out = self(data)

                loss = self.loss_denorm_fn(self.criterion_to_log(out, labels).type(torch.float).cuda().detach())
                valid_loss_l += loss * batch_len
                b += batch_len
            valid_loss = (valid_loss_l / b).type(torch.float).cpu().numpy()[0]
            if not self.mix:
                self.writer.add_scalar("classifier_validation_loss", valid_loss, self.global_step)

            if valid_loss < self.gan_best_valid_loss:
                torch.save(
                    self.state_dict(),
                    os.path.join(self.save_dir,
                                 "checkpoint_best_class_loss.pth"),
                )
                self.gan_best_valid_loss = valid_loss
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
        self.train()

    def run_train(self, loader):
        self.train()
        trainiter = iter(loader)
        self.train_loss_meter = MovingAverageValueMeter(5)
        self.val_loss_meter = MovingAverageValueMeter(5)
        # for iteration in tqdm(
        #         range(1, self.config.ITER_MAX.CLASS + 1),
        #         total=self.config.ITER_MAX.CLASS,
        #         leave=False,
        #         dynamic_ncols=True,
        # ):
        for iteration in range(1,self.config.ITER_MAX.CLASS):
            self.global_step = iteration
            try:
                inputs = trainiter.next()
            except StopIteration:
                trainiter = iter(loader)
                inputs = trainiter.next()
            self.train_step(inputs)

        return self.best_valid_loss

    def initialize_weights(self):
        if self.rnd_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if not m.bias is None:
                        torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if not m.bias is None:
                        torch.nn.init.constant_(m.bias, 0)
        else:
            path = self.config.CLASS_FROZEN_INIT_FILENAME
            self.load_state_dict(torch.load(path), strict=False)

    def get_optimizer(self):
        optimizer = {
            "adam": torch.optim.Adam(
                self.parameters(),
                lr=float(self.config.OPTIMIZER.CLASS.LR_ADAM),
                betas=make_tuple(self.config.OPTIMIZER.CLASS.BETAS),
            )
        }[self.config.OPTIMIZER.CLASS.ALG]
        return optimizer

    def loss_denorm_fn(self, x):
        age_m = make_tuple(self.config.AGE_INTERVAL)[0]
        age_M = make_tuple(self.config.AGE_INTERVAL)[1]
        return x * (age_M - age_m)

    def get_loss_fn(self):

        criterion_to_log = nn.L1Loss().cuda()
        if self.config.ARCHITECTURE.CLASS.LOSS_FN == 'MSE':
            criterion = nn.MSELoss().cuda()
        elif self.config.ARCHITECTURE.CLASS.LOSS_FN == 'L1':
            criterion = nn.L1Loss().cuda()
        elif self.config.ARCHITECTURE.CLASS.LOSS_FN == 'SmoothL1':
            criterion = nn.SmoothL1Loss().cuda()
        else:
            raise NotImplementedError()

        return criterion, criterion_to_log

    def set_dirs(self):
        dataset_name = self.config.DATASET
        if self.mix:
            self.save_dir = os.path.join('save', dataset_name, self.config.run_name, 'mixed')
            self.log_dir = os.path.join('runs', dataset_name, self.config.run_name, 'mixed')
        else:
            self.save_dir = os.path.join('save', dataset_name, self.config.run_name, 'pure')
            self.log_dir = os.path.join('runs', dataset_name, self.config.run_name, 'pure')
        os.makedirs(self.save_dir, exist_ok=True)
