# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 10:37:48 2018

@author: hajduistvan
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

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ACWGAN:
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

        self.netd.init()
        self.netg.init()

        self.global_step = 0
        self.best_valid_loss = np.inf

        self.log_dir = os.path.join('runs', self.config.DATASET, self.config.run_name, 'gan')
        self.save_dir = os.path.join('save', self.config.DATASET, self.config.run_name, 'gan')
        self.writer = SummaryWriter(self.log_dir)

    def train(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE.GAN.TRAIN,
            num_workers=self.config.NUM_WORKERS.GAN,
            shuffle=True,
        )
        dataiter = iter(loader)
        max_iter = self.config.ITER_MAX.GAN
        for iteration in tqdm(
                range(1, max_iter + 1),
                total=max_iter,
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
        if self.global_step % self.config.ITER_TB.GAN == 0:
            self.writer.add_scalar("wasserstein_loss", self.netd.w_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("discriminator_loss", self.netd.d_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("disc_real_critic_loss", self.netd.r_c_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("disc_fake_critic_loss", self.netd.f_c_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("disc_real_regression_loss", self.netd.d_real_regr_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("disc_fake_regression_loss", self.netd.d_fake_regr_loss_meter.value()[0], self.global_step)
            if not self.gan_architecture.LAMBDA_GP == 0:
                self.writer.add_scalar("gradient_penalty", self.netd.gp_loss_meter.value()[0], self.global_step)
            if not self.gan_architecture.LAMBDA_CT == 0:
                self.writer.add_scalar("consistency_cost", self.netd.ct_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("generator_loss", self.netg.g_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("generator_critic_loss", self.netg.g_critic_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("generator_regression_loss", self.netg.g_pred_loss_meter.value()[0], self.global_step)
        if self.global_step % self.config.ITER_SAVE.GAN == 0:
            os.makedirs(self.save_dir, exist_ok=True)
            # torch.save(
            #     self.netd.state_dict(),
            #     os.path.join(self.config.SAVE_DIR.GAN, "checkpoint_disc_{}.pth".format(self.global_step)),
            # )
            # torch.save(
            #     self.netg.state_dict(),
            #     os.path.join(self.config.SAVE_DIR.GAN, "checkpoint_gen_{}.pth".format(self.global_step)),
            # )
            self.netg.visualize_gen_images(self.global_step)
        if self.global_step % self.config.ITER_VAL.GAN == 0:
            self.validate()

    def validate(self):
        gen_data = self.netg.generate_training_images(self.config.GEN_IMG_NUM)
        df = {'train': gen_data}

        gen_dataset = get_dataset(self.config.DATASET)(
            df, config=self.config, split='train')

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
            os.makedirs(self.save_dir, exist_ok=True)
            torch.save(
                self.netd.state_dict(),
                os.path.join(self.save_dir, "checkpoint_disc_best.pth".format(self.global_step)),
            )
            torch.save(
                self.netg.state_dict(),
                os.path.join(self.save_dir, "checkpoint_gen_best.pth".format(self.global_step)),
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
        self.g_loss_meter = MovingAverageValueMeter(5)
        self.g_critic_loss_meter = MovingAverageValueMeter(5)
        self.g_pred_loss_meter = MovingAverageValueMeter(5)
        self.log_dir = os.path.join('runs', self.config.DATASET, self.config.run_name, 'gan')
        self.save_dir = os.path.join('save', self.config.DATASET, self.config.run_name, 'gan')

        self.regr_loss_fn = nn.L1Loss()

        self.preprocess_data = nn.Sequential(
            nn.Linear(self.noise_dim, 2 * self.dim),
            nn.ReLU(True),
        )
        self.fc1_labels = nn.Sequential(
            nn.Linear(1, 2 * self.dim),
            nn.ReLU(True),
        )
        # concat, reshape
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, (1, self.h)),
        )

        self.fc2_labels = nn.Sequential(
            nn.Linear(1, 2 * self.dim * self.h),
        )  # + reshape

        # concat
        self.bn_relu = nn.Sequential(
            nn.BatchNorm2d(4 * self.dim),
            nn.ReLU(True),

        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.dim, 1, (self.h, 1)),  # 2.nd arg DIM
        )
        self.sigmoid = nn.Tanh()
        self.optimizer = self.get_optimizer()

    def forward(self, data, labels):
        labels = labels.view(-1, 1)
        output_data = self.preprocess_data(data)
        output_labels = self.fc1_labels(labels)
        output = torch.cat([output_data, output_labels], 1).view(-1, 4 * self.dim, 1, 1)
        output_data = self.block1(output)
        output_labels = self.fc2_labels(labels).view(-1, 2 * self.dim, 1, self.h)
        output = torch.cat([output_data, output_labels], 1)
        output = self.bn_relu(output)
        output = self.block2(output)
        output = self.sigmoid(output)
        output = (output + torch.transpose(output, -1, -2)) / 2
        return output.view(-1, self.h * self.h)

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if not m.bias is None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if not m.bias is None:
                    torch.nn.init.constant_(m.bias, 0)

    def train_step(self, netd):
        self.zero_grad()

        fake_labels = self.generate_rnd_labels().cuda()

        noise = torch.randn(
            self.config.BATCH_SIZE.GAN.TRAIN,
            self.config.ARCHITECTURE.GAN.NOISE_DIMS
        ).cuda()
        fake = self(noise, fake_labels)
        self.d_fake_critic, self.d_fake_pred = netd(fake)
        self.d_fake_critic = self.d_fake_critic.mean()
        self.fake_regr_loss = self.regr_loss_fn(self.d_fake_pred, fake_labels)
        self.g_cost = - self.d_fake_critic - self.fake_regr_loss
        self.g_cost.backward()
        self.optimizer.step()
        self.hook_fn()

    def hook_fn(self):
        self.g_critic_loss_meter.add(self.d_fake_critic.cpu())
        self.g_pred_loss_meter.add(self.fake_regr_loss.cpu())
        self.g_loss_meter.add(self.g_cost.detach().cpu())

    def get_optimizer(self):
        optimizer = {
            "adam": torch.optim.Adam(
                self.parameters(),
                lr=float(self.config.OPTIMIZER.GAN.G.LR_ADAM),
                betas=make_tuple(self.config.OPTIMIZER.GAN.G.BETAS),
            ),
            # "sgd": torch.optim.SGD(  # Do not use!
            #     self.parameters(),
            #     lr=float(self.config.OPTIMIZER.GAN.G.LR_SGD),
            #     momentum=self.config.OPTIMIZER.GAN.G.MOMENTUM,
            #     nesterov=False,
            # ),
            # "rmsprop": torch.optim.RMSprop(
            #     self.parameters(),
            #     lr=float(self.config.OPTIMIZER.GAN.G.LR_RMS),
            #     momentum=self.config.OPTIMIZER.GAN.G.MOMENTUM,
            # ),
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
        labels = (age_m + (normed_labels + 1) / 2 * (age_M - age_m)).type(torch.int)
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
        return (images, labels)

    def visualize_gen_images(self, global_step):
        num_images = 6
        noise_dims = self.config.ARCHITECTURE.GAN.NOISE_DIMS
        h = self.config.MATR_SIZE
        self.eval()
        noise = torch.randn(num_images, noise_dims).cuda().requires_grad_(False)
        if self.config.SUPERVISE_TYPE == 'binary':
            labels = torch.Tensor([[0], [0], [0],
                                   [1], [1], [1]]).cuda().requires_grad_(False)
            samples = self(noise, labels)
        elif self.config.SUPERVISE_TYPE == 'regress':
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


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        self.config = config
        self.gan_architecture = self.config.ARCHITECTURE.GAN
        self.dim = self.gan_architecture.DIM
        self.h = config.MATR_SIZE
        self.p = self.gan_architecture.DROPOUT
        self.lambda_gp = self.gan_architecture.LAMBDA_GP
        self.lambda_ct = self.gan_architecture.LAMBDA_CT
        self.ct_m = self.gan_architecture.CT_M

        self.w_loss_meter = MovingAverageValueMeter(5)
        self.d_loss_meter = MovingAverageValueMeter(5)
        self.gp_loss_meter = MovingAverageValueMeter(5)
        self.ct_loss_meter = MovingAverageValueMeter(5)
        self.d_real_regr_loss_meter = MovingAverageValueMeter(5)
        self.d_fake_regr_loss_meter = MovingAverageValueMeter(5)
        self.r_c_loss_meter = MovingAverageValueMeter(5)
        self.f_c_loss_meter = MovingAverageValueMeter(5)
        self.regr_loss_fn = nn.L1Loss()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 4 * self.dim, (self.h, 1)),
            nn.ReLU(),
            nn.Dropout2d(self.p)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(4 * self.dim, 2 * self.dim, (1, self.h)),
            nn.ReLU(),
            nn.Dropout(self.p)
        )

        self.block3 = nn.Sequential(
            nn.Linear(2 * self.dim, self.dim),
            nn.ReLU(),
            nn.Dropout(self.p),
        )

        self.critic_output = nn.Linear(self.dim, 1)
        self.prediction_output = nn.Linear(self.dim, 1)

        self.optimizer = self.get_optimizer()

    def forward(self, x):
        x = x.view(-1, 1, self.h, self.h)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x.view(-1, 2 * self.dim))
        out_critic = self.critic_output(x)
        out_pred = self.prediction_output(x)
        return out_critic.view(-1), out_pred.view(-1)

    def init(self):
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

    def get_optimizer(self):
        optimizer = {
            "adam": torch.optim.Adam(
                self.parameters(),
                lr=float(self.config.OPTIMIZER.GAN.D.LR_ADAM),
                betas=make_tuple(self.config.OPTIMIZER.GAN.D.BETAS),
            ),
            # "sgd": torch.optim.SGD(  # Do not use!
            #     self.parameters(),
            #     lr=float(self.config.OPTIMIZER.GAN.D.LR_SGD),
            #     momentum=self.config.OPTIMIZER.GAN.D.MOMENTUM,
            #     nesterov=False,
            # ),
            # "rmsprop": torch.optim.RMSprop(
            #     self.parameters(),
            #     lr=float(self.config.OPTIMIZER.GAN.D.LR_RMS),
            #     momentum=self.config.OPTIMIZER.GAN.D.MOMENTUM,
            # ),
        }[self.config.OPTIMIZER.GAN.D.ALG]
        return optimizer

    def train_step(self, inputs, netg):
        real_data, real_labels = inputs
        real_data = real_data.cuda()
        real_labels = real_labels.cuda()

        self.zero_grad()

        self.d_real_critic, self.d_real_pred = self(real_data)
        self.d_real_critic = self.d_real_critic.mean()

        self.real_regr_loss = self.regr_loss_fn(self.d_real_pred.view(-1,1), real_labels)

        # train with fake
        noise = torch.randn(
            real_data.shape[0],
            self.gan_architecture.NOISE_DIMS
        ).cuda()

        fake = netg(noise, real_labels).data

        self.d_fake_critic, self.d_fake_pred = self(fake)
        self.d_fake_critic = self.d_fake_critic.mean()

        self.fake_regr_loss = self.regr_loss_fn(self.d_fake_pred.view(-1,1), real_labels)

        self.d_cost = self.d_fake_critic - self.d_real_critic + self.real_regr_loss + self.fake_regr_loss
        # train with gradient penalty
        if not self.lambda_gp == 0:
            self.gradient_penalty = self.calc_gradient_penalty_cond(
                real_data.data,
                fake.data
            )
            self.d_cost += self.gradient_penalty * self.lambda_gp

        # consistency_cost
        if not self.lambda_ct == 0:
            self.ct_cost = self.calc_consistency_penalty(
                real_data.data,
                real_labels,
            )
            self.d_cost += self.ct_cost * self.lambda_ct

        self.wasserstein_d = self.d_real_critic - self.d_fake_critic
        self.d_cost.backward()
        self.hook_fn()
        self.optimizer.step()

    def hook_fn(self):

        self.w_loss_meter.add(self.wasserstein_d.detach().cpu())
        self.d_loss_meter.add(self.d_cost.detach().cpu())
        self.d_real_regr_loss_meter.add(self.fake_regr_loss.detach().cpu())
        self.d_fake_regr_loss_meter.add(self.real_regr_loss.detach().cpu())
        self.r_c_loss_meter.add(self.d_real_critic.detach().cpu())
        self.f_c_loss_meter.add(self.d_fake_critic.detach().cpu())

        if not self.lambda_gp == 0:
            self.gp_loss_meter.add(self.gradient_penalty.detach().cpu())
        if not self.lambda_ct == 0:
            self.ct_loss_meter.add(self.ct_cost.detach().cpu())

    def calc_gradient_penalty_cond(self, real_data, fake_data):
        alpha = torch.rand(real_data.size()[0], 1, 1, 1).expand(real_data.size()).cuda()
        interpolates = alpha * real_data + ((1 - alpha) * fake_data.view(-1, 1, self.h, self.h))
        interpolates = interpolates.cuda().requires_grad_(True)

        critic_interpolates, regress_interpolates = self(interpolates)

        gradients = torch.autograd.grad(outputs=[critic_interpolates,regress_interpolates],
                                        inputs=interpolates,
                                        grad_outputs=[torch.ones(critic_interpolates.size()).cuda(),torch.ones(regress_interpolates.size()).cuda()],
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def calc_consistency_penalty(self, real_data, real_labels):
        d1, d_1 = self(real_data, real_labels)
        d2, d_2 = self(real_data, real_labels)

        consistency_term = (d1 - d2).norm(2, dim=0) + 0.1 * (d_1 - d_2).norm(2, dim=1) - self.ct_m
        return consistency_term.mean()
