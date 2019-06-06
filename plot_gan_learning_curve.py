import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
import yaml
from addict import Dict
from models.classifier import ConnectomeConvNet
from data_handling.dataset import UKBioBankDataset, GenDataset
from models.cond_gan import CondGAN
from models.dual_gan import DualGAN


def convert_to_float(df):
    for k, v in df.items():
        if type(v) == str:
            try:
                e = float(v)
            except:
                e = v
            df[k] = e

    return df


class Trainer:
    def __init__(self, config_class, config_gan, gpu_id, log_dir, dataset_root):
        self.config_class = config_class
        self.config_gan = config_gan
        self.log_dir = log_dir
        self.gpu_id = gpu_id
        self.dataset_root = dataset_root
        self.val_interval = 1
        self.test_dataset = UKBioBankDataset(self.dataset_root, None, 'test')
        self.val_dataset = UKBioBankDataset(self.dataset_root, None, 'val')

        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.config_class.batch_size, shuffle=False,
                                                       num_workers=self.config_class.num_workers)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, self.config_class.batch_size, shuffle=False,
                                                      num_workers=self.config_class.num_workers)
        gan_runs_dict = '/home/orthopred/repositories/conn_gan/gan_manual_search/runs/'
        model_cls, model_py = {
            'condgan': (CondGAN, '/home/orthopred/repositories/conn_gan/models/cond_gan.py'),
            'dualgan': (DualGAN, '/home/orthopred/repositories/conn_gan/models/dual_gan.py')
        }[self.config_gan['gan_name']]
        self.gan_model = model_cls(
            self.config_gan,
            self.val_dataset,
            self.val_loader,
            self.gpu_id,
            self.config_gan['fid_interval'],
            '0',
            log_dir,
            self.config_gan['num_epochs'],
        )
        print(self.config_class)
        self.gan_model.netg.load_state_dict(torch.load(os.path.join(gan_runs_dict, self.config_class.run_name, 'ckpts',
                                                                    'ckpt_0_step_' + str(
                                                                        self.config_class.gan_step) + '_disc.pth'))[
                                                'netg'])

    def set_train_loader(self, mode_str, num_total_examples):
        if mode_str == 'real':
            real_dataset = UKBioBankDataset(self.dataset_root, num_total_examples, 'train')
            self.train_loader = torch.utils.data.DataLoader(real_dataset, self.config_class.batch_size, shuffle=True,
                                                            num_workers=self.config_class.num_workers)
        elif mode_str == 'gen':
            imgs, labels = self.gan_model.netg.generate_fake_images(num_total_examples)
            gen_dataset = GenDataset(imgs.cpu(), labels.cpu())
            self.train_loader = torch.utils.data.DataLoader(gen_dataset, self.config_class.batch_size, shuffle=True,
                                                            num_workers=self.config_class.num_workers)
        elif mode_str == 'r+g':
            real_dataset = UKBioBankDataset(self.dataset_root, num_total_examples // 2, 'train')
            imgs, labels = self.gan_model.netg.generate_fake_images(num_total_examples // 2)
            gen_dataset = GenDataset(imgs.cpu(), labels.cpu())
            mixed_dataset = torch.utils.data.ConcatDataset([real_dataset, gen_dataset])
            self.train_loader = torch.utils.data.DataLoader(mixed_dataset, self.config_class.batch_size, shuffle=True,
                                                            num_workers=self.config_class.num_workers)

        if mode_str == 'r+z':
            real_dataset = UKBioBankDataset(self.dataset_root, num_total_examples, 'train', random_noise_hardening=True)
            self.train_loader = torch.utils.data.DataLoader(real_dataset, self.config_class.batch_size, shuffle=True,
                                                            num_workers=self.config_class.num_workers)
        print('Dataset built.')

    def run_train_cycle(self, mode_str, numbers):
        """

        :param mode_str: 'real': only real examples. 'gen': only gan-generated examples. 'r+g': real + gen 1:1. 'r+z': real + noisy data 1:1
        :param numbers: total number of examples
        :return:
        """
        test_losses, test_accs = [], []
        for num_training_examples in numbers:
            self.set_train_loader(mode_str, num_training_examples)

            classifier = ConnectomeConvNet(
                (self.config_class.c1, self.config_class.c2),
                self.config_class.lr,
                self.config_class.mom,
                self.config_class.wd,
                self.train_loader,
                self.val_loader,
                self.gpu_id,
                self.val_interval,
                0,
                os.path.join(self.log_dir, mode_str + str(num_training_examples)),
                self.config_class.max_epochs,
                allow_stop=False,
                verbose=True
            )
            loss, acc, num_params = classifier.run_train()
            testloss, testacc = classifier.test(self.test_loader)
            print(testloss, testacc)
            test_losses.append(testloss)
            test_accs.append(test_accs)
        return test_losses, test_accs


def plot_learning_curves(args):
    cfg_dir = '/home/orthopred/repositories/conn_gan/config'
    runs_dir = '/home/orthopred/repositories/conn_gan/learning_curve_plots'
    log_dir = os.path.join(runs_dir, args.exp_name)
    config_class = Dict(yaml.load(open(os.path.join(cfg_dir, args.config_class))))
    config_gan = convert_to_float(
        yaml.load(open('/home/orthopred/repositories/conn_gan/gan_manual_search/runs/cond_gan_debug_24/config.yaml')))
    trainer = Trainer(config_class, config_gan, args.gpu_id, log_dir, args.dataset_root)
    mode_str = 'real'
    losses, accs = trainer.run_train_cycle(mode_str, config_class.num_training_examples)

    plt.plot(config_class.num_training_examples, losses)
    plt.xlabel('Number of real training examples')
    plt.ylabel('Test loss')
    plt.title('Test loss with respect to available training examples')
    plt.savefig(os.path.join(log_dir, 'testlosses.png'))
    plt.savefig(os.path.join(log_dir, 'testlosses.eps'))
    plt.close('all')
    plt.plot(config_class.num_training_examples, accs)
    plt.xlabel('Number of real training examples')
    plt.ylabel('Test accuracy')
    plt.title('Test accuracy with respect to available training examples')
    plt.savefig(os.path.join(log_dir, 'testaccs.png'))
    plt.savefig(os.path.join(log_dir, 'testaccs.eps'))
    plt.close('all')


def get_gan_testloss_metrics():
    # steps = [0, 50, 100, 150, 174]
    # steps = [25, 75, 125, 180]
    # steps = [280, 320]
    steps = [254, 300]
    cfg_dir = '/home/orthopred/repositories/conn_gan/config'

    dataset_root = '/home/orthopred/repositories/Datasets/UK_Biobank'
    config_class=Dict(yaml.load(open(os.path.join(cfg_dir, 'plot_lr_curve_cfg.yaml'))))
    test_dataset = UKBioBankDataset(dataset_root, None, 'test')
    val_dataset = UKBioBankDataset(dataset_root, None, 'val')

    test_loader = torch.utils.data.DataLoader(test_dataset, config_class.batch_size, shuffle=False,
                                                   num_workers=config_class.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, config_class.batch_size, shuffle=False,
                                                  num_workers=config_class.num_workers)

    log_dir = '/home/orthopred/repositories/conn_gan/gan_manual_search/runs/cond_gan_debug_25/'
    os.makedirs(log_dir, exist_ok=True)
    dir = '/home/orthopred/repositories/conn_gan/gan_manual_search/runs/cond_gan_debug_24/ckpts'
    gan_model = CondGAN(
        convert_to_float(
            yaml.load(
                open('/home/orthopred/repositories/conn_gan/gan_manual_search/runs/cond_gan_debug_24/config.yaml'))),
        val_dataset,
        val_loader,
        1,
        1,
        '0',
        log_dir,
        200,
    )
    test_losses, test_accs = [], []

    for step in steps:
        fname = os.path.join(dir, 'ckpt_0_step_'+str(step)+'_disc.pth')
        gan_model.netg.load_state_dict(torch.load(fname)['netg'])
        imgs, labels = gan_model.netg.generate_fake_images(14861)
        gen_dataset = GenDataset(imgs.cpu(), labels.cpu())
        train_loader = torch.utils.data.DataLoader(gen_dataset, 14861, shuffle=True,
                                                        num_workers=0)

        classifier = ConnectomeConvNet(
            (config_class.c1, config_class.c2),
            config_class.lr,
            config_class.mom,
            config_class.wd,
            train_loader,
            val_loader,
            1,
            1,
            0,
            os.path.join(log_dir, 'gen' + str(step)),
            200,
            allow_stop=False,
            verbose=True
        )
        loss, acc, num_params = classifier.run_train()
        testloss, testacc = classifier.test(test_loader)
        print(step, testloss, testacc)
        test_losses.append(testloss)
        test_accs.append(test_accs)
    # np.save(os.path.join(log_dir, 'loss_result.npy'), np.array([steps, test_losses, test_accs]))

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--exp_name", type=str, default='debug_real_100')
    # parser.add_argument("--config_class", type=str, default='plot_lr_curve_cfg.yaml')
    #
    # parser.add_argument("--gpu_id", type=int, default=1)
    # parser.add_argument("--dataset_root", type=str, default='/home/orthopred/repositories/Datasets/UK_Biobank')
    # args = parser.parse_args()
    #
    # plot_learning_curves(args)
    get_gan_testloss_metrics()
