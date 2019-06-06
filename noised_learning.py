import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
import yaml
from addict import Dict
from models.classifier import ConnectomeConvNet
from data_handling.dataset import UKBioBankDataset


def convert_to_float(df):
    for k, v in df.items():
        if type(v) == str:
            try:
                e = float(v)
            except:
                e = v
            df[k] = e

    return df


class NoisedTrainer:
    def __init__(self, gpu_id, log_dir, dataset_root):
        cfg = '/home/orthopred/repositories/conn_gan/config/plot_lr_curve_cfg.yaml'
        self.config_class = Dict(yaml.load(open(cfg)))
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

    def set_train_loader(self, aug_fn, num_total_examples=14864):
            real_dataset = UKBioBankDataset(self.dataset_root, num_total_examples, 'train', aug_fn=aug_fn)
            self.train_loader = torch.utils.data.DataLoader(real_dataset, self.config_class.batch_size, shuffle=True,
                                                            num_workers=self.config_class.num_workers)

    def run_train_cycle(self, run_id):
        """

        :param mode_str: 'real': only real examples. 'gen': only gan-generated examples. 'r+g': real + gen 1:1. 'r+z': real + noisy data 1:1
        :param numbers: total number of examples
        :return:
        """

        classifier = ConnectomeConvNet(
            (self.config_class.c1, self.config_class.c2),
            self.config_class.lr,
            self.config_class.mom,
            self.config_class.wd,
            self.train_loader,
            self.val_loader,
            self.gpu_id,
            self.val_interval,
            run_id,
            os.path.join(self.log_dir, str(run_id)),
            self.config_class.max_epochs,
            allow_stop=False,
            verbose=True
        )
        _,_,_ = classifier.run_train()
        testloss, testacc = classifier.test(self.test_loader)
        print(testloss, testacc)
        return testloss, testacc

# def plot_learning_curves(args):
#     cfg_dir = '/home/orthopred/repositories/conn_gan/config'
#     runs_dir = '/home/orthopred/repositories/conn_gan/learning_curve_plots'
#     log_dir = os.path.join(runs_dir, args.exp_name)
#     config_class = Dict(yaml.load(open(os.path.join(cfg_dir, args.config_class))))
#     config_gan = convert_to_float(
#         yaml.load(open('/home/orthopred/repositories/conn_gan/gan_manual_search/runs/cond_gan_debug_12/config.yaml')))
#     trainer = Trainer(config_class, config_gan, args.gpu_id, log_dir, args.dataset_root)
#     mode_str = 'real'
#     losses, accs = trainer.run_train_cycle(mode_str, config_class.num_training_examples)
#
#
#     plt.plot(config_class.num_training_examples, losses)
#     plt.xlabel('Number of real training examples')
#     plt.ylabel('Test loss')
#     plt.title('Test loss with respect to available training examples')
#     plt.savefig(os.path.join(log_dir, 'testlosses.png'))
#     plt.savefig(os.path.join(log_dir, 'testlosses.eps'))
#     plt.close('all')
#     plt.plot(config_class.num_training_examples, accs)
#     plt.xlabel('Number of real training examples')
#     plt.ylabel('Test accuracy')
#     plt.title('Test accuracy with respect to available training examples')
#     plt.savefig(os.path.join(log_dir, 'testaccs.png'))
#     plt.savefig(os.path.join(log_dir, 'testaccs.eps'))
#     plt.close('all')

