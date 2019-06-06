# -*- coding: utf-8 -*-
"""
@author István Hajdu at MTA TTK
https://github.com/hajduistvan/connectome_gan
"""
import torch
import torch.nn as nn
from tqdm import tqdm
from ast import literal_eval as make_tuple
import numpy as np
import os
import matplotlib
from torchnet.meter import MovingAverageValueMeter
from tensorboardX import SummaryWriter
from gan_metrics.calc_metrics import MetricCalculator
from gan_metrics.select_cnn import get_model

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DualGAN:
    def __init__(
            self,
            hyperparameters,
            train_dataset,
            val_loader,
            gpu_id,
            fid_interval,
            run_id,
            log_dir,
            max_epochs,
    ):

        # constants
        self.tensorboard_log_interval = 1  # global_step
        self.metric_model_id = 1187

        self.train_dataset = train_dataset
        self.hyperparameters = hyperparameters
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.hyperparameters['batch_size'], shuffle=True, )
        self.val_loader = val_loader
        self.gpu_id = gpu_id
        self.fid_interval = fid_interval
        self.run_id = run_id
        self.log_dir = log_dir

        self.numpy_log = {
            "step": [],
            'wasserstein_loss': [],
            "discriminator_loss": [],
            "disc_real_loss": [],
            "disc_fake_loss": [],
            "gradient_penalty": [],
            "generator_ce_loss": [],
            "generator_loss": [],
            "gen_gp": [],
            "gan_fid_score": [],
            "gan_fid_c0": [],
            "gan_fid_c1": [],
            "gan_fid_c_mean": [],
            "gan_ce_loss": [],
            "gan_bin_distance": [],
            "gan_bin_c0": [],
            "gan_bin_c1": [],
            "gan_bin_c_mean": [],
        }
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'scatterplots', self.run_id), exist_ok=True)

        self.metric_calculator = MetricCalculator(self.metric_model_id, self.train_dataset,
                                                  self.hyperparameters['batch_size_metric'],
                                                  self.gpu_id)
        self.max_epochs = max_epochs
        self.netg = Generator(self.hyperparameters, self.log_dir, self.gpu_id, self.run_id)
        self.netd = Discriminator(self.hyperparameters, self.gpu_id)

        self.global_step = 0
        self.critic_iters = self.hyperparameters['critic_iters']
        os.makedirs(os.path.join(self.log_dir, 'ckpts'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'summaries'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'vis_imgs'), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.log_dir, 'summaries'))

    def run_train(self):
        for ep in tqdm(
                range(self.max_epochs),
                total=self.max_epochs,
                leave=False,
                dynamic_ncols=True,
        ):
            dataiter = iter(self.train_loader)
            for inputs in dataiter:
                for p in self.netd.parameters():
                    p.requires_grad = True
                # Train critic for N steps
                for iter_d in range(self.critic_iters):
                    self.netd.train_step(inputs, self.netg)
                # Disable grad accumulation for critic
                for p in self.netd.parameters():
                    p.requires_grad = False
                # Train generator network
                self.netg.train_step(self.netd)

                self.global_hook_fn()
                self.global_step += 1

    def global_hook_fn(self):
        if self.global_step % self.tensorboard_log_interval == 0:
            self.writer.add_scalar("wasserstein_loss", self.netd.w_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("discriminator_loss", self.netd.d_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("disc_real_loss", self.netd.r_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("disc_fake_loss", self.netd.f_loss_meter.value()[0], self.global_step)
            if not self.hyperparameters['lambda_gp'] == 0:
                self.writer.add_scalar("gradient_penalty", self.netd.gp_loss_meter.value()[0], self.global_step)
            if not self.hyperparameters['lambda_ce'] == 0:
                self.writer.add_scalar("generator_ce_loss", self.netg.g_ce_loss_meter.value()[0], self.global_step)
            if not self.hyperparameters['lambda_g_gp'] == 0:
                self.writer.add_scalar("generator_gradient_penalty", self.netg.g_gp_cost_meter.value()[0],
                                       self.global_step)
            self.writer.add_scalar("generator_loss", self.netg.g_loss_meter.value()[0], self.global_step)
        if self.global_step % self.fid_interval == 0:
            self.calc_metrics()
            self.netg.visualize_gen_images(self.global_step)

    def update_numpy_log(self):
        self.numpy_log["step"].append(self.global_step)
        self.numpy_log['wasserstein_loss'].append(self.netd.wasserstein_d.detach().cpu().numpy())
        self.numpy_log["discriminator_loss"].append(self.netd.d_cost.detach().cpu().numpy())
        self.numpy_log["disc_real_loss"].append(self.netd.d_real.detach().cpu().numpy())
        self.numpy_log["disc_fake_loss"].append(self.netd.d_fake.detach().cpu().numpy())
        if self.hyperparameters['lambda_gp'] != 0:
            self.numpy_log["gradient_penalty"].append(self.netd.gradient_penalty.detach().cpu().numpy())
        if self.hyperparameters['lambda_g_gp'] != 0:

            self.numpy_log["generator_ce_loss"].append(self.netg.gen_ce_loss.detach().cpu().numpy())
        self.numpy_log["generator_loss"].append(self.netg.g_cost.detach().cpu().numpy())
        self.numpy_log["gan_fid_score"].append(self.fid_score)
        self.numpy_log["gan_fid_c0"].append(self.fid_c0)
        self.numpy_log["gan_fid_c1"].append(self.fid_c1)
        self.numpy_log["gan_fid_c_mean"].append(self.fid_c_mean)
        self.numpy_log["gan_ce_loss"].append(self.ce_loss)
        self.numpy_log["gan_bin_distance"].append(self.bin_dist_score)
        self.numpy_log["gan_bin_c0"].append(self.bin_dist_score_c0)
        self.numpy_log["gan_bin_c1"].append(self.bin_dist_score_c1)
        self.numpy_log["gan_bin_c_mean"].append(self.bin_dist_score_c_mean)
        if self.hyperparameters['lambda_g_gp'] != 0:
            self.numpy_log['gen_gp'].append(self.netg.gen_gp_cost)

    def calc_metrics(self):
        with torch.no_grad():
            gen_batch, gen_labels = self.netg.generate_fake_images(self.hyperparameters['batch_size_metric'])
            self.metric_calculator.feed_batch(gen_batch, gen_labels)
            self.fid_score = self.metric_calculator.calc_fid()
            self.fid_c0, self.fid_c1, self.fid_c_mean = self.metric_calculator.calc_class_agnostic_fid()
            self.bin_dist_score = self.metric_calculator.calc_bin_dist()
            self.bin_dist_score_c0, self.bin_dist_score_c1, self.bin_dist_score_c_mean, = self.metric_calculator.calc_bin_class_agnostic()

            self.ce_loss = self.metric_calculator.calc_crossentropy()
            self.metric_calculator.scatter_plot_activations(
                os.path.join(self.log_dir, 'scatterplots', self.run_id, 'gen_img_it_' + str(self.global_step) + '.svg'))
        self.writer.add_scalar("gan_fid_score", self.fid_score, self.global_step)
        self.writer.add_scalar("gan_fid_c0", self.fid_c0, self.global_step)
        self.writer.add_scalar("gan_fid_c1", self.fid_c1, self.global_step)
        self.writer.add_scalar("gan_fid_c_mean", self.fid_c_mean, self.global_step)

        self.writer.add_scalar("gan_ce_loss", self.ce_loss, self.global_step)
        self.writer.add_scalar("gan_bin_distance", self.bin_dist_score, self.global_step)
        self.writer.add_scalar("gan_bin_c0", self.bin_dist_score_c0, self.global_step)
        self.writer.add_scalar("gan_bin_c1", self.bin_dist_score_c1, self.global_step)
        self.writer.add_scalar("gan_bin_c_mean", self.bin_dist_score_c_mean, self.global_step)
        self.update_numpy_log()
        torch.save(
            {"netd": self.netd.state_dict(),
             "netg": self.netg.state_dict(),
             'optd': self.netd.optimizer.state_dict(),
             'optg': self.netg.optimizer.state_dict(),
             'numpy_log': self.numpy_log
             },
            os.path.join(self.log_dir, 'ckpts',
                         "ckpt_" + self.run_id + '_step_' + str(self.global_step) + "_disc.pth"),
        )


class Generator(nn.Module):
    def __init__(self, hyperparameters, log_dir, gpu_id, run_id):
        super(Generator, self).__init__()
        self.hyp = hyperparameters
        print(hyperparameters)
        self.gpu_id = gpu_id
        self.run_id = run_id
        self.noise_dim = self.hyp['noise_dim']
        self.vis_noise = torch.randn(1, self.hyp['noise_dim']).cuda(self.gpu_id).requires_grad_(False)
        self.g_loss_meter = MovingAverageValueMeter(5)
        self.g_ce_loss_meter = MovingAverageValueMeter(5)
        self.g_gp_cost_meter = MovingAverageValueMeter(5)
        self.log_dir = log_dir
        np.save(os.path.join(self.log_dir, 'vis_noise.npy'), self.vis_noise.cpu().numpy())
        self.inceptor_module = get_model(self.gpu_id, 1187)
        for p in self.inceptor_module.parameters():
            p.requires_grad = False

        self.conv0 = nn.ConvTranspose2d(2 * self.noise_dim, self.hyp['p1'], (55, 1), groups=2, bias=False)
        self.nonlin0 = nn.Sequential(
            *[nn.BatchNorm2d(self.hyp['p1'] + self.hyp['p2']), nn.LeakyReLU(self.hyp['lrelu_g'])] if self.hyp[
                'bg0'] else [nn.LeakyReLU(self.hyp['lrelu_g']), ])

        self.conv1 = nn.ConvTranspose2d(self.hyp['p1'], 1, (1, 55))
        self.sigmoid = nn.Tanh()
        self.cuda(self.gpu_id)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hyp['lr_g'],
                                          betas=(self.hyp['b1_g'], self.hyp['b2_g']), weight_decay=self.hyp['wd_g'])
        # milestones = [self.config.ITER_MAX.GAN // 10 * s for s in range(10)]
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
        #                                                       gamma=self.config.OPTIMIZER.GAN.G.POWER)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, a=self.hyp['lrelu_g'], nonlinearity='leaky_relu')
                if not m.bias is None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, z, labels):
        z = z.view(-1, self.noise_dim, 1, 1)
        labels = labels.view(-1, 1, 1, 1).float()
        x = torch.cat([z * labels, z * (1 - labels)], 1)
        x = self.conv0(x)
        x = self.nonlin0(x)
        x = self.conv1(x)
        x = (x + torch.transpose(x, -1, -2)) / 2
        x = self.sigmoid(x)

        return x

    def train_step(self, netd):
        self.zero_grad()

        self.fake_labels = torch.randint(0, 2, (self.hyp['batch_size'],)).type(torch.float).cuda(self.gpu_id)

        self.noise = torch.randn(
            self.hyp['batch_size'],
            self.hyp['noise_dim']).cuda(self.gpu_id)
        self.g = self(self.noise, self.fake_labels)
        self.g_cost = -netd(self.g, self.fake_labels).mean()
        if self.hyp['lambda_ce'] != 0:
            self.gen_ce_loss = self.calc_gen_ce_loss()
            self.g_cost += self.hyp['lambda_ce'] * self.gen_ce_loss
            self.g_ce_loss_meter.add(self.gen_ce_loss.detach().cpu())
        if self.hyp['lambda_g_gp'] != 0:
            self.gen_gp_cost = self.calc_generator_grad_penalty()
            self.g_cost += self.hyp['lambda_g_gp'] * self.gen_gp_cost
        self.g_cost.backward()
        self.optimizer.step()
        self.g_loss_meter.add(self.g_cost.detach().cpu())
        if self.hyp['lambda_g_gp'] != 0:
            self.g_gp_cost_meter.add(self.gen_gp_cost.detach().cpu())

    def calc_generator_grad_penalty(self):
        self.noise.requires_grad_(True)
        self.g.requires_grad_(True)
        self.fake_labels.requires_grad_(True)
        gradients = torch.autograd.grad(outputs=self.g,
                                        inputs=[self.noise, self.fake_labels],
                                        grad_outputs=None,  # torch.ones(self.g.size()).cuda(self.gpu_id),
                                        create_graph=True, retain_graph=True, only_inputs=True,
                                        allow_unused=True)[0]
        print('gen grads: ', gradients)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def calc_gen_ce_loss(self):
        _, _, out = self.inceptor_module(self.g)
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(out.view(-1), self.fake_labels,
                                                                       reduction='none')
        target = torch.ones_like(ce_loss) * 0.37205103
        loss = torch.nn.functional.mse_loss(ce_loss, target)
        return loss

    def generate_fake_images(self, num_images):
        self.eval()
        labels = (torch.randint(0, 2, (num_images,))).type(torch.long).cuda(self.gpu_id)
        noise = torch.randn(num_images, self.hyp['noise_dim']).cuda(self.gpu_id).requires_grad_(False)
        images = self(noise, labels).detach()
        self.train()
        return images, labels

    def visualize_gen_images(self, global_step):
        self.eval()

        noise = torch.cat([self.vis_noise, self.vis_noise], 0)

        labels = (torch.from_numpy(np.array([0, 1]))).type(torch.long).view(-1, 1).cuda(
            self.gpu_id).requires_grad_(False)
        samples = self(noise, labels)

        i = str(global_step)
        os.makedirs(os.path.join(self.log_dir, 'vis_imgs', self.run_id), exist_ok=True)
        filename = os.path.join(self.log_dir, 'vis_imgs', self.run_id, 'gen_img_it_' + i)
        b, chs, h, w = samples.shape
        imgs = samples.view(b, h, w).detach().cpu().data.numpy()
        np.save(filename + '.npy', imgs)
        labels = labels.view(b).detach().cpu().data.numpy()
        fig = plt.figure()
        for i in range(b):
            plt.subplot(1, 2, i + 1)
            plt.imshow(imgs[i], cmap='jet', interpolation='nearest', vmin=-1, vmax=1)
            plt.title('Sex: ' + ['Female', 'Male'][labels[i]])
            plt.axis('off')
        plt.savefig(filename + '.eps')
        plt.savefig(filename + '.png')
        plt.close()
        self.train()


class Discriminator(nn.Module):
    def __init__(self, hyperparameters, gpu_id):
        super(Discriminator, self).__init__()

        self.hyp = hyperparameters
        self.gpu_id = gpu_id
        self.w_loss_meter = MovingAverageValueMeter(5)
        self.d_loss_meter = MovingAverageValueMeter(5)
        self.r_loss_meter = MovingAverageValueMeter(5)
        self.f_loss_meter = MovingAverageValueMeter(5)
        self.gp_loss_meter = MovingAverageValueMeter(5)

        self.conv0 = nn.Conv2d(2, self.hyp['q1'], (55, 1), groups=2, bias=False)
        self.nonlin0 = nn.Sequential(
            *[nn.BatchNorm2d(self.hyp['q1']), nn.LeakyReLU(self.hyp['lrelu_d'])] if self.hyp[
                'bd0'] else [nn.LeakyReLU(self.hyp['lrelu_d']), ])

        self.conv1 = nn.Conv2d(self.hyp['q1'], self.hyp['q2'], (1, 55), groups=2, bias=False)
        self.nonlin1 = nn.Sequential(
            *[nn.BatchNorm2d(self.hyp['q2']), nn.LeakyReLU(self.hyp['lrelu_d'])] if self.hyp['bd1'] else [
                nn.LeakyReLU(self.hyp['lrelu_d']), ])

        self.fc = nn.Linear(self.hyp['q2'], 1, bias=False)

        self.cuda(self.gpu_id)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hyp['lr_d'],
                                          betas=(self.hyp['b1_d'], self.hyp['b2_d']), weight_decay=self.hyp['wd_d'])
        # milestones = [self.config.ITER_MAX.GAN // 10 * s for s in range(10)]
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
        #                                                       gamma=self.config.OPTIMIZER.GAN.D.POWER)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, a=self.hyp['lrelu_d'], nonlinearity='leaky_relu')
                if not m.bias is None:
                    torch.nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, a=self.hyp['lrelu_d'], nonlinearity='leaky_relu')
                if not m.bias is None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if not m.bias is None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x, l):
        x = x.view(-1, 1, 55, 55)
        labels = l.view(-1, 1, 1, 1).float()
        x = torch.cat([x * labels, x * (1 - labels)], 1)
        x = self.conv0(x)
        x = self.nonlin0(x)
        x = self.conv1(x)
        x = self.nonlin1(x)
        x = x.view(-1, self.hyp['q2'])
        x = self.fc(x)
        return x

    def train_step(self, inputs, netg):
        real_data, real_labels = inputs
        real_data = real_data.cuda(self.gpu_id)
        real_labels = real_labels.cuda(self.gpu_id)

        self.zero_grad()

        self.d_real = self(real_data, real_labels).mean()

        # train with fake
        noise = torch.randn(
            real_data.shape[0],
            self.hyp['noise_dim']
        ).cuda(self.gpu_id)

        fake = netg(noise, real_labels).data

        self.d_fake = self(fake, real_labels).mean()

        self.d_cost = self.d_fake - self.d_real
        # train with gradient penalty
        if not self.hyp['lambda_gp'] == 0:
            self.gradient_penalty = self.calc_gradient_penalty_cond(
                real_data.data,
                real_labels,
                fake.data
            )
            self.d_cost += self.gradient_penalty * self.hyp['lambda_gp']

        self.wasserstein_d = self.d_real - self.d_fake
        self.d_cost.backward()
        self.optimizer.step()

        self.w_loss_meter.add(self.wasserstein_d.detach().cpu())
        self.d_loss_meter.add(self.d_cost.detach().cpu())
        self.r_loss_meter.add(self.d_real.detach().cpu())
        self.f_loss_meter.add(self.d_fake.detach().cpu())

        if not self.hyp['lambda_gp'] == 0:
            self.gp_loss_meter.add(self.gradient_penalty.detach().cpu())

    def calc_gradient_penalty_cond(self, real_data, real_labels, fake_data):
        alpha = torch.rand(real_data.size()[0], 1, 1, 1).expand(real_data.size()).cuda(self.gpu_id)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.cuda(self.gpu_id).requires_grad_(True)

        real_labels.requires_grad_(True)
        disc_interpolates = self(interpolates, real_labels)

        gradients = torch.autograd.grad(outputs=disc_interpolates,
                                        inputs=[interpolates, real_labels],
                                        grad_outputs=torch.ones(disc_interpolates.size()).cuda(self.gpu_id),
                                        create_graph=True, retain_graph=True, only_inputs=True,
                                        allow_unused=True)[0]
        # print('critic grads: ', gradients)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
