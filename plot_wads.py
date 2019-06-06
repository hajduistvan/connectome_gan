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
import yaml
from data_handling.dataset import UKBioBankDataset
from models.cond_gan import CondGAN
import os
import matplotlib.pyplot as plt
from torchnet.meter import MovingAverageValueMeter
from tensorboardX import SummaryWriter
from gan_metrics.calc_metrics import MetricCalculator
from gan_metrics.select_cnn import get_model

dot_steps = [0, 50, 100, 150, 174]


def convert_to_float(df):
    for k, v in df.items():
        if type(v) == str:
            try:
                e = float(v)
            except:
                e = v
            df[k] = e

    return df


def plot_stuff():
    dir = '/home/orthopred/repositories/conn_gan/gan_manual_search/runs/cond_gan_debug_24/ckpts'
    dir2 = '/home/orthopred/repositories/conn_gan/gan_manual_search/runs/cond_gan_debug_24/wad_plots'
    cfg_file = '/home/orthopred/repositories/conn_gan/gan_manual_search/runs/cond_gan_debug_24/config.yaml'
    cfg = convert_to_float(yaml.load(open(cfg_file)))
    train_dataset = UKBioBankDataset('/home/orthopred/repositories/Datasets/UK_Biobank', None, 'train')

    val_loader = torch.utils.data.DataLoader(train_dataset, cfg['batch_size'], shuffle=False,
                                             num_workers=cfg['num_workers'])
    model = CondGAN(cfg, train_dataset, val_loader, 0, 10, '0', dir2, 200)

    # Datasets & Loaders

    os.makedirs(dir2, exist_ok=True)
    fnames = ['ckpt_0_step_' + str(i) + '_disc.pth' for i in range(0, 181)]
    steps = []
    wads = []
    png_name = os.path.join(dir2, 'wad_plot.')

    for i, f in enumerate(fnames):
        print(i)
        fname = os.path.join(dir, f)
        gen_w = torch.load(fname)['netg']
        model.netg.load_state_dict(gen_w)
        with torch.no_grad():
            gen_batch, gen_labels = model.netg.generate_fake_images(model.hyperparameters['batch_size_metric'])
            model.metric_calculator.feed_batch(gen_batch.detach(), gen_labels.detach())
            _, _, fid_c_mean = model.metric_calculator.calc_class_agnostic_fid()
        steps.append(i)
        wads.append(fid_c_mean)
    print(steps, wads)
    plt.plot(steps, wads)
    plt.plot(dot_steps, np.array(wads)[dot_steps], 'r.')
    plt.grid(which='both', axis='both')

    # plt.xlim(left=0, right=181)
    plt.yscale('log')
    plt.title('WAD with respect to iteration number')
    plt.xlabel('Iteration')
    plt.ylabel('WAD')
    plt.savefig(png_name + 'png')
    plt.savefig(png_name + 'svg')
    plt.savefig(png_name + 'eps')
    plt.close('all')


def plot_wasserstein_stuff():
    dir = '/home/orthopred/repositories/conn_gan/gan_manual_search/runs/cond_gan_debug_24/ckpts'
    dir2 = '/home/orthopred/repositories/conn_gan/gan_manual_search/runs/cond_gan_debug_24/wad_plots'
    cfg_file = '/home/orthopred/repositories/conn_gan/gan_manual_search/runs/cond_gan_debug_24/config.yaml'
    cfg = convert_to_float(yaml.load(open(cfg_file)))
    train_dataset = UKBioBankDataset('/home/orthopred/repositories/Datasets/UK_Biobank', None, 'train')

    val_loader = torch.utils.data.DataLoader(train_dataset, cfg['batch_size'], shuffle=False,
                                             num_workers=cfg['num_workers'])
    model = CondGAN(cfg, train_dataset, val_loader, 0, 10, '0', dir2, 200)

    # Datasets & Loaders

    os.makedirs(dir2, exist_ok=True)
    fname = 'ckpt_0_step_' + str(181) + '_disc.pth'
    png_name = os.path.join(dir2, 'wass_plot.')

    npy_log = torch.load(os.path.join(dir, fname))['numpy_log']
    w_loss = npy_log['wasserstein_loss']
    steps = npy_log['step']

    print(steps, w_loss)
    plt.plot(steps, w_loss)
    plt.plot(dot_steps, np.array(w_loss)[dot_steps], 'r.')
    plt.grid(which='both', axis='both')
    # plt.xlim(left=0, right=181)
    plt.yscale('log')
    plt.title('Wasserstein Loss with respect to iteration number')
    plt.xlabel('Iteration')
    plt.ylabel('Wasserstein Loss')
    plt.savefig(png_name + 'png')
    plt.savefig(png_name + 'svg')
    plt.savefig(png_name + 'eps')
    plt.close('all')


def plot_gen_imgs():
    dir = '/home/orthopred/repositories/conn_gan/gan_manual_search/runs/cond_gan_debug_24/vis_imgs/0'
    dir2 = '/home/orthopred/repositories/conn_gan/gan_manual_search/runs/cond_gan_debug_24/gen_imgs'
    os.makedirs(dir2, exist_ok=True)
    fnames = ['gen_img_it_' + str(i) + '.npy' for i in range(0, 181)]

    # real example
    train_dataset = UKBioBankDataset('/home/orthopred/repositories/Datasets/UK_Biobank', None, 'train')
    loader = torch.utils.data.DataLoader(train_dataset, 200, shuffle=True,
                                         num_workers=0)
    iterloader = iter(loader)
    batch = next(iterloader)
    female_examples = batch[0][(batch[1] == 0).view(-1), 0, :, :][0,:,:]
    male_examples = batch[0][(batch[1] == 1).view(-1), 0, :, :][0,:,:]
    real_examples = [female_examples, male_examples]
    fig = plt.figure()
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(real_examples[i], cmap='jet', interpolation='nearest', vmin=-1, vmax=1)
        plt.title('Sex: ' + ['Female', 'Male'][i])
        plt.axis('off')
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
        #                     hspace=1, wspace=0.1)
        # plt.margins(0.1, 0.1)
    plt.savefig(os.path.join(dir2, 'real_example') + '.eps', bbox_inches='tight', pad_inches=0.0)
    plt.savefig(os.path.join(dir2, 'real_example') + '.png', bbox_inches='tight', pad_inches=0.0)
    plt.close()

    # # generated examples
    # for j, fname in enumerate(fnames):
    #     fname = os.path.join(dir, fname)
    #     img_fname = os.path.join(dir2, 'gen_'+str(j))
    #     arr = np.load(fname)
    #     fig = plt.figure()
    #     for i in range(2):
    #         plt.subplot(1, 2, i + 1)
    #         plt.imshow(arr[i,:,:], cmap='jet', interpolation='nearest', vmin=-1, vmax=1)
    #         plt.title('Sex: ' + ['Female', 'Male'][i])
    #         plt.axis('off')
    #     plt.savefig(img_fname + '.eps', bbox_inches='tight', pad_inches=0.0)
    #     plt.savefig(img_fname + '.png', bbox_inches='tight', pad_inches=0.0)
    #     plt.close()


# plot_stuff()
# plot_wasserstein_stuff()
plot_gen_imgs()
