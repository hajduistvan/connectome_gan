# -*- coding: utf-8 -*-
"""
@author István Hajdu at MTA TTK
https://github.com/hajduistvan/connectome_gan
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ast import literal_eval as make_tuple
import numpy as np
import os
import matplotlib
from torchnet.meter import MovingAverageValueMeter
from tensorboardX import SummaryWriter
from libs.datasets import get_dataset
from libs.models.convnet import ConnectomeConvNet
from libs.utils.coordconv import CoordConv, CoordConvTranspose, pad_kernel_init

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AugNet:
    def __init__(
            self,
            config,
            train_dataset,
            val_loader
    ):
        self.config = config
        self.train_dataset = train_dataset
        self.val_loader = val_loader

        self.h = config.MATR_SIZE
        self.gan_architecture = self.config.ARCHITECTURE.GAN

        self.dim = self.gan_architecture.DIM
        self.noise_dims = self.gan_architecture.NOISE_DIMS
        self.p = self.gan_architecture.DROPOUT
        self.critic_iters = self.gan_architecture.CRITIC_ITERS

        self.netg = Generator(self.config)
        self.netd = Discriminator(self.config)

        self.global_step = 0
        self.best_valid_loss = np.inf
        self.critic_iters = self.config.ARCHITECTURE.GAN.CRITIC_ITERS
        self.max_iter = self.config.ITER_MAX.GAN
        if self.config.ARCHITECTURE.GAN.CRITIC_ITER_DECAY:
            self.critic_iters_milestones = [int(i/self.critic_iters*self.max_iter) for i in range(1, self.critic_iters)]
        self.log_dir = os.path.join('runs', self.config.DATASET, self.config.RUN_NAME)

        self.writer = SummaryWriter(self.log_dir)

    def train(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE.GAN.TRAIN,
            num_workers=self.config.NUM_WORKERS.GAN,
            shuffle=True,
        )
        dataiter = iter(loader)

        for iteration in tqdm(
                range(1, self.max_iter + 1),
                total=self.max_iter,
                leave=False,
                dynamic_ncols=True,
        ):
            self.global_step = iteration
            # Enable grad accumulation for critic
            for p in self.netd.parameters():
                p.requires_grad = True
            # Train critic for N steps
            for iter_d in range(self.critic_iters):
                try:
                    inputs = dataiter.next()
                except StopIteration:
                    dataiter = iter(loader)
                    inputs = dataiter.next()
                self.netd.train_step(inputs, self.netg)
            # Disable grad accumulation for critic
            for p in self.netd.parameters():
                p.requires_grad = False
            # Train generator network
            self.netg.train_step(self.netd)

            self.global_hook_fn()
        return self.best_valid_loss

    def global_hook_fn(self):
        self.netg.scheduler.step()
        self.netd.scheduler.step()
        if self.global_step % self.config.ITER_TB.GAN == 0:
            self.writer.add_scalar("disc_real_regression_loss", self.netd.d_real_regr_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("disc_fake_regression_loss", self.netd.d_fake_regr_loss_meter.value()[0], self.global_step)
            if not self.gan_architecture.LAMBDA_GP == 0:
                self.writer.add_scalar("gradient_penalty", self.netd.gp_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("generator_regression_loss", self.netg.g_regr_loss_meter.value()[0], self.global_step)
        if self.global_step % self.config.ITER_SAVE.GAN == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            self.netg.visualize_gen_images(self.global_step)
        if self.global_step % self.config.ITER_VAL.GAN == 0:
            self.validate()
        if self.config.ARCHITECTURE.GAN.CRITIC_ITER_DECAY and self.global_step in self.critic_iters_milestones:
            self.critic_iters -= self.config.ARCHITECTURE.GAN.CRITIC_ITER_DECAY
    def validate(self):
        gen_data = self.netg.generate_training_images(self.config.GEN_IMG_NUM)


        gen_dataset = get_dataset(self.config.DATASET)(
            self.config, split='train', gen_values=gen_data)

        mixed_dataset = torch.utils.data.ConcatDataset([self.train_dataset, gen_dataset])
        mixed_loader = torch.utils.data.DataLoader(
            dataset=mixed_dataset,
            batch_size=self.config.BATCH_SIZE.CLASS.TRAIN,
            num_workers=self.config.NUM_WORKERS.CLASS,
            shuffle=True,
            pin_memory=True
        )
        cnn = ConnectomeConvNet(self.config, True, self.val_loader, gan_best_valid_loss=self.best_valid_loss).cuda()
        val_loss = cnn.run_train(mixed_loader)
        self.writer.add_scalar("gan_valid_loss", val_loss, self.global_step)

        if val_loss < self.best_valid_loss:
            self.best_valid_loss = val_loss
            os.makedirs(self.log_dir, exist_ok=True)
            torch.save(
                self.netd.state_dict(),
                os.path.join(self.log_dir, "checkpoint_disc_best.pth".format(self.global_step)),
            )
            torch.save(
                self.netg.state_dict(),
                os.path.join(self.log_dir, "checkpoint_gen_best.pth".format(self.global_step)),
            )


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.config = config
        self.gan_architecture = self.config.ARCHITECTURE.GAN
        self.dim = self.gan_architecture.DIM
        self.noise_dim = self.gan_architecture.NOISE_DIMS
        self.h = config.MATR_SIZE
        self.noise_dims = self.gan_architecture.NOISE_DIMS
        self.use_coord = make_tuple(self.gan_architecture.COORDCONV_G)
        self.a = float(self.gan_architecture.LRELU_SLOPE)
        self.a = 0
        self.g_regr_loss_meter = MovingAverageValueMeter(5)
        self.log_dir = os.path.join('runs', self.config.DATASET, self.config.RUN_NAME, 'gan')

        self.regr_loss_fn = nn.L1Loss()
        # Model
        self.lab_0 = nn.Sequential(
            nn.Conv2d(1, self.dim, 1)
        )
        self.block0 = nn.Sequential(
            nn.Conv2d(self.noise_dim, 3 * self.dim, 1),
        )

        self.block_lab_0 = nn.Sequential(
            nn.BatchNorm2d(self.dim * 4),
            nn.LeakyReLU(negative_slope=self.a, inplace=True),
        )

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, (1, self.h)),
            nn.BatchNorm2d(self.dim * 2),
            nn.LeakyReLU(negative_slope=self.a, inplace=True),
        )
        self.block2 = nn.Sequential(
            CoordConvTranspose(2 * self.dim, self.dim, use_coord=self.use_coord[0], kernel_size=(self.h, 1)),  # 2.nd arg DIM
            nn.BatchNorm2d(self.dim),
            nn.LeakyReLU(negative_slope=self.a, inplace=True),
        )
        self.block3 = nn.Sequential(
            CoordConv(self.dim, 1, use_coord=self.use_coord[1], kernel_size=1),
            nn.BatchNorm2d(1),
        )
        self.sigmoid = nn.Tanh()

        ###

        self.optimizer = self.get_optimizer()
        milestones = [self.config.ITER_MAX.GAN // 10 * s for s in range(10)]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                              gamma=self.config.OPTIMIZER.GAN.G.POWER)

        self.init_weights()

    def forward(self, x, labels):
        x = x.view(-1, self.noise_dim, 1, 1)
        labels = labels.view(-1, 1, 1, 1)
        x = torch.cat([self.block0(x), self.lab_0(labels)], 1)
        x = self.block_lab_0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.sigmoid(x)
        x = (x+torch.transpose(x, -1, -2))/2
        return x.view(-1, self.h, self.h)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, a=self.a, nonlinearity='leaky_relu')
                if not m.bias is None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, a=self.a, nonlinearity='leaky_relu')
                if not m.bias is None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, CoordConvTranspose):
                torch.nn.init.kaiming_normal_(m.conv.weight, a=self.a, nonlinearity='leaky_relu')
                if not m.conv.bias is None:
                    torch.nn.init.constant_(m.conv.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
    def train_step(self, netd):
        self.zero_grad()

        fake_labels = self.generate_rnd_labels().cuda()

        noise = torch.randn(
            self.config.BATCH_SIZE.GAN.TRAIN,
            self.config.ARCHITECTURE.GAN.NOISE_DIMS
        ).cuda()
        fake = self(noise, fake_labels)
        self.d_fake_regr = netd(fake)
        self.fake_regr_loss = self.regr_loss_fn(self.d_fake_regr, fake_labels)
        self.fake_regr_loss.backward()
        self.optimizer.step()
        self.hook_fn()

    def hook_fn(self):
        self.g_regr_loss_meter.add(self.loss_denorm_fn(self.fake_regr_loss.detach().cpu()))

    def get_optimizer(self):
        optimizer = {
            "adam": torch.optim.Adam(
                self.parameters(),
                lr=float(self.config.OPTIMIZER.GAN.G.LR_ADAM),
                betas=make_tuple(self.config.OPTIMIZER.GAN.G.BETAS),
            ),
        }[self.config.OPTIMIZER.GAN.G.ALG]
        return optimizer

    def generate_rnd_labels(self):
        batch_size = self.config.BATCH_SIZE.GAN.TRAIN
        return (torch.rand(batch_size) * 2 - 1).type(torch.float)

    def generate_training_images(self, num_images):
        self.eval()
        b = self.config.BATCH_SIZE.GAN.GENERATE
        noise_dims = self.config.ARCHITECTURE.GAN.NOISE_DIMS

        age_m = make_tuple(self.config.AGE_INTERVAL)[0]
        age_M = make_tuple(self.config.AGE_INTERVAL)[1]
        normed_labels = (torch.rand(num_images) * 2 - 1).cuda()
        labels = (age_m + (normed_labels + 1) / 2 * (age_M - age_m))
        images = np.empty((num_images, 1, self.h, self.h)).astype(np.float32)
        # for i in tqdm(
        #         range(0, num_images // b),
        #         total=num_images // b,
        #         leave=False,
        #         dynamic_ncols=True,
        # ):
        for i in range(num_images // b):
            noise = torch.randn(b, noise_dims).cuda().requires_grad_(False)
            samples = self(noise, normed_labels[i * b:(i + 1) * b])
            samples = samples.view(b, 1, self.h, self.h).detach().cpu().data.numpy().astype(np.float32)
            images[i * b:(i + 1) * b] = samples
        labels = labels.cpu().numpy().reshape(-1)
        self.train()
        return images, labels

    def visualize_gen_images(self, global_step):
        num_images = 6
        noise_dims = self.config.ARCHITECTURE.GAN.NOISE_DIMS
        h = self.config.MATR_SIZE
        self.eval()
        noise = torch.randn(num_images, noise_dims).cuda().requires_grad_(False)
        age_m = make_tuple(self.config.AGE_INTERVAL)[0]
        age_M = make_tuple(self.config.AGE_INTERVAL)[1]
        lab1 = (age_m + torch.rand(num_images // 2) * (55 - age_m)).type(torch.int)
        lab2 = (55 + torch.rand(num_images // 2) * (age_M - 55)).type(torch.int)
        labels = torch.cat([lab1, lab2], 0).type(torch.float).cuda().requires_grad_(False)
        samples = self(noise, (labels - (age_M - age_m) / 2) / (age_M - age_m))
        samples = samples.view(num_images, 1, h, h)

        i = str(global_step)
        os.makedirs(self.log_dir, exist_ok=True)
        filename = os.path.join(self.log_dir, 'gen_img_it_' + i + '.png')
        b, chs, h, w = samples.shape
        imgs = samples.view(b, h, w).detach().cpu().data.numpy()
        labels = labels.view(b).detach().cpu().data.numpy()
        fig = plt.figure()
        for i in range(b):
            plt.subplot(2, 3, i + 1)
            plt.imshow(imgs[i], cmap='jet', interpolation='nearest', vmin=-1, vmax=1)
            plt.title('Age: ' + str(labels[i]))
            plt.axis('off')
        plt.savefig(filename)
        self.train()
    def loss_denorm_fn(self, x):
        age_m = make_tuple(self.config.AGE_INTERVAL)[0]
        age_M = make_tuple(self.config.AGE_INTERVAL)[1]
        return x * (age_M - age_m)

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.config = config
        self.gan_architecture = self.config.ARCHITECTURE.GAN
        self.dim = self.gan_architecture.DIM
        self.h = config.MATR_SIZE
        self.p = self.gan_architecture.DROPOUT
        self.lambda_gp = self.gan_architecture.LAMBDA_GP
        self.a = float(self.gan_architecture.LRELU_SLOPE)
        self.use_coord = make_tuple(self.gan_architecture.COORDCONV_G)

        self.d_loss_meter = MovingAverageValueMeter(5)
        self.gp_loss_meter = MovingAverageValueMeter(5)
        self.d_real_regr_loss_meter = MovingAverageValueMeter(5)
        self.d_fake_regr_loss_meter = MovingAverageValueMeter(5)
        self.regr_loss_fn = nn.L1Loss()

        self.block0 = nn.Sequential(
            CoordConv(1, 8 * self.dim, use_coord=self.use_coord[0], kernel_size=(self.h, 1)),
            # nn.BatchNorm2d(4 * self.dim),
            nn.LeakyReLU(negative_slope=self.a),
            # nn.Dropout2d(self.p)
        )

        self.block1 = nn.Sequential(
            CoordConv(8 * self.dim, 4 * self.dim, use_coord=self.use_coord[1], kernel_size=(1, self.h)),
            # nn.BatchNorm2d(2 * self.dim),
            nn.LeakyReLU(negative_slope=self.a),
            # nn.Dropout2d(self.p)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(4 * self.dim, 2 * self.dim, 1),
            # nn.BatchNorm2d(self.dim),
            nn.LeakyReLU(negative_slope=self.a),
            # nn.Dropout2d(self.p),
        )
        self.last_common = nn.Sequential(
            nn.Linear(2 * self.dim, 2 * self.dim),
            nn.LeakyReLU(negative_slope=self.a),
        )
        self.regr_fc = nn.Sequential(
            nn.Linear(2 * self.dim, self.dim),
            nn.LeakyReLU(negative_slope=self.a),
        )
        self.prediction_output = nn.Linear(self.dim, 1)

        self.optimizer = self.get_optimizer()
        milestones = [self.config.ITER_MAX.GAN // 10 * s for s in range(10)]
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                              gamma=self.config.OPTIMIZER.GAN.D.POWER)


        self.init_weights()

    def forward(self, x):
        x = self.block0(x.view(-1, 1, self.h, self.h))
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(-1, 2 * self.dim)
        x = self.last_common(x)
        x_regr = self.regr_fc(x)
        x_pred = self.prediction_output(x_regr)
        return x_pred.view(-1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, a=self.a, nonlinearity='leaky_relu')
                if not m.bias is None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=self.a, nonlinearity='leaky_relu')
                if not m.bias is None:
                    torch.nn.init.constant_(m.bias, 0)


    def get_optimizer(self):
        optimizer = {
            "adam": torch.optim.Adam(
                self.parameters(),
                lr=float(self.config.OPTIMIZER.GAN.D.LR_ADAM),
                betas=make_tuple(self.config.OPTIMIZER.GAN.D.BETAS),
            ),
        }[self.config.OPTIMIZER.GAN.D.ALG]
        return optimizer

    def train_step(self, inputs, netg):
        real_data, real_labels = inputs
        real_data = real_data.cuda()
        real_labels = real_labels.cuda()

        self.zero_grad()

        self.d_real_pred = self(real_data)

        self.real_regr_loss = self.regr_loss_fn(self.d_real_pred.view(-1,1), real_labels)

        # train with fake
        noise = torch.randn(
            real_data.shape[0],
            self.gan_architecture.NOISE_DIMS
        ).cuda()

        fake = netg(noise, real_labels).data

        self.d_fake_pred = self(fake)
        self.fake_regr_loss = self.regr_loss_fn(self.d_fake_pred.view(-1,1), real_labels)

        self.d_cost = self.real_regr_loss
        if not self.lambda_gp == 0:
            self.gradient_penalty = self.calc_gradient_penalty_cond(
                real_data.data,
                fake.data
            )
            self.d_cost += self.gradient_penalty * self.lambda_gp


        self.d_cost.backward()
        self.hook_fn()
        self.optimizer.step()

    def hook_fn(self):

        self.d_loss_meter.add(self.d_cost.detach().cpu())
        self.d_real_regr_loss_meter.add(self.loss_denorm_fn(self.real_regr_loss.detach().cpu()))
        self.d_fake_regr_loss_meter.add(self.loss_denorm_fn(self.fake_regr_loss.detach().cpu()))

        if not self.lambda_gp == 0:
            self.gp_loss_meter.add(self.gradient_penalty.detach().cpu())

    def calc_gradient_penalty_cond(self, real_data, fake_data):
        alpha = torch.rand(real_data.size()[0], 1, 1, 1).expand(real_data.size()).cuda()
        interpolates = alpha * real_data + ((1 - alpha) * fake_data.view(-1, 1, self.h, self.h))
        interpolates = interpolates.cuda().requires_grad_(True)

        regress_interpolates = self(interpolates)

        gradients = torch.autograd.grad(outputs=[regress_interpolates],
                                        inputs=interpolates,
                                        grad_outputs=[torch.ones(regress_interpolates.size()).cuda()],
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def loss_denorm_fn(self, x):
        age_m = make_tuple(self.config.AGE_INTERVAL)[0]
        age_M = make_tuple(self.config.AGE_INTERVAL)[1]
        if isinstance(self.regr_loss_fn, nn.MSELoss):
            x = torch.sqrt(x)
        return x * (age_M - age_m)