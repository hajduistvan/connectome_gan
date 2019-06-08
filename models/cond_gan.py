# -*- coding: utf-8 -*-
"""
@author István Hajdu at MTA TTK
https://github.com/hajduistvan/connectome_gan
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
import matplotlib
from torchnet.meter import MovingAverageValueMeter
from tensorboardX import SummaryWriter
from gan_metrics.calc_metrics import MetricCalculator

matplotlib.use('Agg')
import matplotlib.pyplot as plt


class CondGAN:
    def __init__(
            self,
            hyperparameters,
            train_dataset,
            val_loader,
            gpu_id,
            wad_interval,
            log_dir,
            max_epochs,
            cnn_arch_dir,
            metric_model_id,
    ):
        """

        :param hyperparameters: Dict of hyperparameters, for generator, discriminator and optimizer.
        :param train_dataset:
        :param val_loader:
        :param gpu_id:
        :param wad_interval: how many steps to pass between WAD calculations
        :param log_dir: directory where the checkpoints and tensorboard sumamries are saved.
        :param max_epochs: stop training after this.
        :param cnn_arch_dir: directory of trained CNNS (for reference network loading)
        """
        # constants
        self.tensorboard_log_interval = 1  # global_step
        self.metric_model_id = metric_model_id

        self.train_dataset = train_dataset
        self.hyperparameters = hyperparameters
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.hyperparameters['batch_size'], shuffle=True, )
        self.val_loader = val_loader
        self.gpu_id = gpu_id
        self.wad_interval = wad_interval
        self.log_dir = log_dir
        self.cnn_arch_dir = cnn_arch_dir
        self.numpy_log = {
            "step": [],
            'wasserstein_loss': [],
            "discriminator_loss": [],
            "disc_real_loss": [],
            "disc_fake_loss": [],
            "gradient_penalty": [],
            "generator_loss": [],
            "wad": [],
        }
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'scatterplots'), exist_ok=True)

        self.metric_calculator = MetricCalculator(self.metric_model_id, self.train_dataset,
                                                  self.hyperparameters['batch_size_metric'],
                                                  self.gpu_id, self.cnn_arch_dir)
        self.max_epochs = max_epochs
        self.netg = Generator(self.hyperparameters, self.log_dir, self.gpu_id)
        self.netd = Discriminator(self.hyperparameters, self.gpu_id)

        self.global_step = 0
        self.critic_iters = self.hyperparameters['critic_iters']
        os.makedirs(os.path.join(self.log_dir, 'ckpts'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'summaries'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'vis_imgs'), exist_ok=True)
        self.writer = SummaryWriter(os.path.join(self.log_dir, 'summaries'))

    def run_train(self):
        """
        Runs the training procedure.
        :return:
        """
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
        """
        Logs scalars and generated images, reference net activations.
        """
        if self.global_step % self.tensorboard_log_interval == 0:
            self.writer.add_scalar("wasserstein_loss", self.netd.w_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("discriminator_loss", self.netd.d_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("disc_real_loss", self.netd.r_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("disc_fake_loss", self.netd.f_loss_meter.value()[0], self.global_step)
            if not self.hyperparameters['lambda_gp'] == 0:
                self.writer.add_scalar("gradient_penalty", self.netd.gp_loss_meter.value()[0], self.global_step)
            self.writer.add_scalar("generator_loss", self.netg.g_loss_meter.value()[0], self.global_step)
        if self.global_step % self.wad_interval == 0:
            self.calc_metrics()
            self.netg.visualize_gen_images(self.global_step)

    def update_numpy_log(self):
        """
        Logs scalars to numpy files for future plotting and stuff.
        :return:
        """
        self.numpy_log["step"].append(self.global_step)
        self.numpy_log['wasserstein_loss'].append(self.netd.wasserstein_d.detach().cpu().numpy())
        self.numpy_log["discriminator_loss"].append(self.netd.d_cost.detach().cpu().numpy())
        self.numpy_log["disc_real_loss"].append(self.netd.d_real.detach().cpu().numpy())
        self.numpy_log["disc_fake_loss"].append(self.netd.d_fake.detach().cpu().numpy())
        if self.hyperparameters['lambda_gp'] != 0:
            self.numpy_log["gradient_penalty"].append(self.netd.gradient_penalty.detach().cpu().numpy())
        self.numpy_log["generator_loss"].append(self.netg.g_cost.detach().cpu().numpy())
        self.numpy_log["wad"].append(self.wad)

    def calc_metrics(self):
        """
        Calculates WAD, saves activations to a scatter plot.
        :return:
        """
        with torch.no_grad():
            gen_batch, gen_labels = self.netg.generate_fake_images(self.hyperparameters['batch_size_metric'])
            self.metric_calculator.feed_batch(gen_batch, gen_labels)
            self.wad = self.metric_calculator.calc_wad()
            self.metric_calculator.scatter_plot_activations(
                os.path.join(self.log_dir, 'scatterplots', 'gen_img_it_' + str(self.global_step) + '.svg'))
        self.writer.add_scalar("wad", self.wad, self.global_step)
        self.update_numpy_log()
        torch.save(
            {"netd": self.netd.state_dict(),
             "netg": self.netg.state_dict(),
             'optd': self.netd.optimizer.state_dict(),
             'optg': self.netg.optimizer.state_dict(),
             'numpy_log': self.numpy_log
             },
            os.path.join(self.log_dir, 'ckpts',
                         "ckpt" +'_step_' + str(self.global_step) + ".pth"),
        )


class Generator(nn.Module):
    def __init__(self, hyperparameters, log_dir, gpu_id):
        super(Generator, self).__init__()
        self.hyp = hyperparameters
        print(hyperparameters)
        self.gpu_id = gpu_id
        self.noise_dim = self.hyp['noise_dim']
        self.vis_noise = torch.randn(1, self.hyp['noise_dim']).cuda(self.gpu_id).requires_grad_(False)
        self.g_loss_meter = MovingAverageValueMeter(5)
        self.log_dir = log_dir

        # Architecture:
        self.lab0 = nn.Linear(1, self.hyp['p1'], bias=False)
        self.fc0 = nn.Linear(self.noise_dim, self.hyp['p2'], bias=False)
        self.nonlin0 = nn.Sequential(
            *[nn.BatchNorm2d(self.hyp['p1'] + self.hyp['p2']), nn.LeakyReLU(self.hyp['lrelu_g'])] if self.hyp[
                'bg0'] else [nn.LeakyReLU(self.hyp['lrelu_g']), ])

        self.conv1 = nn.ConvTranspose2d(self.hyp['p1'] + self.hyp['p2'], self.hyp['p3'], (1, 55), bias=True)
        self.nonlin1 = nn.Sequential(
            *[nn.BatchNorm2d(self.hyp['p3']), nn.LeakyReLU(self.hyp['lrelu_g'])] if self.hyp['bg1'] else [
                nn.LeakyReLU(self.hyp['lrelu_g']), ])

        self.conv2 = nn.ConvTranspose2d(self.hyp['p3'], 1, (55, 1), bias=True)
        self.sigmoid = nn.Tanh()

        self.cuda(self.gpu_id)
        opt_param_list = [{'params': [param for name, param in self.named_parameters() if 'lab0' not in name]},
                          {'params': self.lab0.parameters(), 'lr': 1 * self.hyp['lr_g']}]

        self.optimizer = torch.optim.Adam(opt_param_list, lr=self.hyp['lr_g'],
                                          betas=(self.hyp['b1_g'], self.hyp['b2_g']), weight_decay=self.hyp['wd_g'])
        # rand init
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight, a=self.hyp['lrelu_g'], nonlinearity='leaky_relu')
                if not m.bias is None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if not m.bias is None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, z, labels):
        x = z.view(-1, self.noise_dim)
        labels = labels.view(-1, 1).float() * 2 - 1
        x = torch.cat([self.fc0(x).view(-1, self.hyp['p2'], 1, 1), self.lab0(labels).view(-1, self.hyp['p1'], 1, 1)], 1)
        x = self.nonlin0(x)
        x = self.conv1(x)
        x = self.nonlin1(x)
        x = self.conv2(x)
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
        self.g_cost.backward()
        self.optimizer.step()
        self.g_loss_meter.add(self.g_cost.detach().cpu())


    def generate_fake_images(self, num_images):
        self.eval()
        labels = (torch.randint(0, 2, (num_images,))).type(torch.long).cuda(self.gpu_id)
        noise = torch.randn(num_images, self.hyp['noise_dim']).cuda(self.gpu_id).requires_grad_(False)
        images = self(noise, labels).detach()
        self.train()
        return (images, labels)

    def visualize_gen_images(self, global_step):
        """
        Saves sample of generated images to a eps and png file. Note that the noise input of
        the generator for visualizing is the same during training.
        :param global_step:
        :return:
        """
        self.eval()

        noise = torch.cat([self.vis_noise, self.vis_noise], 0)

        labels = (torch.from_numpy(np.array([0, 1]))).type(torch.long).view(-1, 1).cuda(
            self.gpu_id).requires_grad_(False)
        samples = self(noise, labels)

        i = str(global_step)
        os.makedirs(os.path.join(self.log_dir, 'vis_imgs'), exist_ok=True)
        filename = os.path.join(self.log_dir, 'vis_imgs', 'gen_img_it_' + i)
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

        # Architecture
        self.lab0 = nn.ConvTranspose2d(1, self.hyp['q1'], (1, 55), bias=False)
        self.conv0 = nn.Conv2d(1, self.hyp['q2'], (55, 1), bias=False)
        self.nonlin0 = nn.Sequential(
            *[nn.BatchNorm2d(self.hyp['q1'] + self.hyp['q2']), nn.LeakyReLU(self.hyp['lrelu_d'])] if self.hyp[
                'bd0'] else [nn.LeakyReLU(self.hyp['lrelu_d']), ])

        self.conv1 = nn.Conv2d(self.hyp['q1'] + self.hyp['q2'], self.hyp['q3'], (1, 55), bias=False)
        self.nonlin1 = nn.Sequential(
            *[nn.BatchNorm2d(self.hyp['q3']), nn.LeakyReLU(self.hyp['lrelu_d'])] if self.hyp['bd1'] else [
                nn.LeakyReLU(self.hyp['lrelu_d']), ])

        self.fc = nn.Linear(self.hyp['q3'], 1, bias=False)

        self.cuda(self.gpu_id)
        opt_param_list = [{'params': [param for name, param in self.named_parameters() if 'lab0' not in name]},
                          {'params': self.lab0.parameters(), 'lr': 1 * self.hyp['lr_d']}]

        self.optimizer = torch.optim.Adam(opt_param_list, lr=self.hyp['lr_d'],
                                          betas=(self.hyp['b1_d'], self.hyp['b2_d']), weight_decay=self.hyp['wd_d'])
        # rand init
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
        x = self.conv0(x)
        l = self.lab0(l.float().view(-1, 1, 1, 1)) * 2 - 1
        x = torch.cat([x, l], 1)
        x = self.nonlin0(x)
        x = self.conv1(x)
        x = self.nonlin1(x)
        x = x.view(-1, self.hyp['q3'])
        x = self.fc(x)
        return x

    def train_step(self, inputs, netg):
        """
        One training step.
        :param inputs:
        :param netg:
        :return:
        """
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
        """
        Calculates Gradient Penalty.
        :param real_data:
        :param real_labels:
        :param fake_data:
        :return:
        """
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

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
