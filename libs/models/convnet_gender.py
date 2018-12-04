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
from libs.utils.coordconv import CoordConv, pad_kernel_init


class ConnectomeConvNet(nn.Module):

    def __init__(
            self,
            config,
            val_loader,
            exp_name,
            gan_best_valid_loss=np.inf,
            rnd_init=False,

    ):
        super(ConnectomeConvNet, self).__init__()
        self.config = config
        self.class_architecture = self.config.ARCHITECTURE.CLASS
        self.exp_name = exp_name
        self.val_loader = val_loader

        dataset_name = self.config.DATASET
        self.log_dir = os.path.join('runs', dataset_name, self.config.RUN_NAME)
        os.makedirs(self.log_dir, exist_ok=True)
        self.h = self.config.MATR_SIZE
        self.rnd_init = rnd_init
        self.chs = make_tuple(self.class_architecture.CHS)
        self.neurons = make_tuple(self.class_architecture.NEURONS)
        self.p = self.class_architecture.DROPOUT
        self.coord_convs = make_tuple(self.class_architecture.COORDCONV)
        self.best_valid_loss = np.inf
        self.best_valid_acc = 0
        self.gan_best_valid_loss = gan_best_valid_loss
        self.criterion = nn.CrossEntropyLoss()

        self.writer = SummaryWriter(self.log_dir)

        self.block1 = nn.Sequential(
            # CoordConv(1, self.chs[0], use_coord=self.coord_convs[0], kernel_size=(self.h, 1)),
            nn.Conv2d(1,self.chs[0], kernel_size=(self.h, 1)),
            nn.BatchNorm2d(self.chs[0]),
            nn.ReLU(),
            nn.Dropout2d(self.p),
        )
        self.block2 = nn.Sequential(
            # CoordConv(self.chs[0], self.chs[1], use_coord=self.coord_convs[1], kernel_size=(1, self.h)),
            nn.Conv2d(self.chs[0], self.chs[1], kernel_size=(1,self.h)),
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
        milestones = [self.config.ITER_MAX.CLASS // 10 * s for s in range(10)]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                              gamma=self.config.OPTIMIZER.CLASS.POWER)

        self.initialize_weights()

    def forward(self, x):

        x = self.block1(x)#.view(-1,1,self.h,self.h))
        x = self.block2(x)
        x = x.view(-1, self.chs[1])
        x = self.block3(x)
        x = self.block4(x)
        x = self.out(x)
        return x

    def train_step(self, inputs):
        self.scheduler.step()
        data, labels = inputs
        data = data.cuda()
        labels = labels.view(-1).cuda()
        self.optimizer.zero_grad()

        outputs = self(data)
        self.loss = self.criterion(outputs, labels)
        self.accuracy = self.get_acc(outputs.detach(), labels)
        self.loss.backward()
        self.optimizer.step()
        self.hook_fn()

    def hook_fn(self):
        self.train_loss_meter.add(self.loss.cpu().detach())
        self.train_acc_meter.add(self.accuracy)

        if self.global_step % self.config.ITER_TB.CLASS == 0:
            self.writer.add_scalar("classifier_training_loss", self.train_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("classifier_training_accuracy", self.train_acc_meter.value()[0], self.global_step)

        if self.global_step % self.config.ITER_VAL.CLASS == 0:
            self.validate()

    def get_acc(self, outputs, labels):
        """
        Operates on detached cuda tensors!
        :param outputs:
        :param labels:
        :return:
        """
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total
        # print('pred: ', predicted)
        # print('labels: ', labels)
        # print('               ')
        # print('correct: ', correct)
        # print('acc: ', accuracy)
        # print('______________________')


        return accuracy

    def validate(self):
        self.eval()
        valid_loss_l = torch.cuda.FloatTensor(1).fill_(0)
        valid_acc_l = torch.cuda.FloatTensor(1).fill_(0)
        b = 0
        with torch.no_grad():
            for i, datapair in enumerate(self.val_loader):
                data, labels = datapair
                batch_len = data.shape[0]
                labels = labels.type(torch.long).view(-1)
                data, labels = data.cuda(), labels.cuda()
                out = self(data)
                loss = self.criterion(out, labels).type(torch.float).cuda().detach()
                acc = self.get_acc(out.detach(), labels)
                valid_loss_l += loss * batch_len
                valid_acc_l += acc * batch_len
                b += batch_len
            valid_loss = (valid_loss_l / b).type(torch.float).cpu().numpy()[0]
            valid_acc = (valid_acc_l / b).type(torch.float).cpu().numpy()[0]

            self.writer.add_scalar("classifier_validation_loss", valid_loss, self.global_step)
            self.writer.add_scalar("classifier_validation_accuracy", valid_acc, self.global_step)

            if valid_loss < self.gan_best_valid_loss:
                torch.save(
                    self.state_dict(),
                    os.path.join(self.log_dir,
                                 "checkpoint_best_class_"+self.exp_name+".pth"),
                )
                self.gan_best_valid_loss = valid_loss
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.best_valid_acc = valid_acc

        self.train()

    def run_train(self, loader):
        self.train()
        trainiter = iter(loader)
        self.train_loss_meter = MovingAverageValueMeter(5)
        self.val_loss_meter = MovingAverageValueMeter(5)
        self.train_acc_meter = MovingAverageValueMeter(5)
        self.val_acc_meter = MovingAverageValueMeter(5)
        for iteration in tqdm(
                range(1, self.config.ITER_MAX.CLASS + 1),
                total=self.config.ITER_MAX.CLASS,
                leave=False,
                dynamic_ncols=True,
        ):
            # for iteration in range(1,self.config.ITER_MAX.CLASS):
            self.global_step = iteration
            try:
                inputs = trainiter.next()
            except StopIteration:
                trainiter = iter(loader)
                inputs = trainiter.next()
            self.train_step(inputs)

        return self.best_valid_loss, self.best_valid_acc

    def initialize_weights(self):
        if self.rnd_init:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if not m.bias is None:
                        torch.nn.init.constant_(m.bias, 0)
                elif isinstance(m, CoordConv):
                    # torch.nn.init.kaiming_normal_(m.conv.weight, nonlinearity='relu')
                    torch.nn.init.normal_(m.conv.weight)
                    if not m.conv.bias is None:
                        torch.nn.init.constant_(m.conv.bias, 0)
                    # m.conv.weight[:, -2:, :, :] = 1e-10
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
            path = os.path.join('frozen_inits',self.config.CLASS_FROZEN_INIT_FILENAME)
            state_dict = torch.load(path)
            # print(list(state_dict.keys()))
            #
            # if self.coord_convs[0]:
            #     state_dict['block1.0.conv.weight'] = pad_kernel_init(state_dict['block1.0.conv.weight'])
            # if self.coord_convs[1]:
            #     state_dict['block2.0.conv.weight'] = pad_kernel_init(state_dict['block2.0.conv.weight'])
            self.load_state_dict(state_dict, strict=False)

    def get_optimizer(self):
        optimizer = {
            "adam": torch.optim.Adam(
                self.parameters(),
                lr=float(self.config.OPTIMIZER.CLASS.LR_ADAM),
                betas=make_tuple(self.config.OPTIMIZER.CLASS.BETAS),
                weight_decay=float(self.config.OPTIMIZER.CLASS.WD)
            )
        }[self.config.OPTIMIZER.CLASS.ALG]
        return optimizer

    def evaluate(self, test_loader, path=None):
        if path is None:
            path = os.path.join(self.log_dir, "checkpoint_best_class_"+self.exp_name+".pth")
        self.eval()
        self.load_state_dict(torch.load(path))
        l, a, b = 0, 0, 0
        for i, datapair in enumerate(test_loader):
            with torch.no_grad():
                data, labels = datapair
                batch_len = data.shape[0]
                data = data.cuda()
                labels = labels.type(torch.long).view(-1).cuda()
                output = self(data)
                acc = self.get_acc(output.detach(),labels)
                loss = self.criterion(output, labels)
                ll = loss.detach().cpu().numpy()

                l += ll * batch_len
                a+=acc*batch_len
                b += batch_len
        testloss = l / b
        testacc = a / b
        self.writer.add_scalar('test_loss', testloss)
        self.writer.add_scalar('test_acc', testacc)
        # print('Test loss: ', testloss)
        return testloss, testacc
