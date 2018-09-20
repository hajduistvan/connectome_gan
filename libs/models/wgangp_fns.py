# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:04:09 2018

@author: hajduistvan
"""
import torch
import numpy as np
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import os
from ast import literal_eval as make_tuple


def calc_gradient_penalty_cond(netD, real_data, real_labels, fake_data):
    alpha = torch.rand(real_data.size()[0], 1, 1, 1).expand(real_data.size()).cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda().requires_grad_(True)

    real_labels.requires_grad_(True)
    disc_interpolates, _ = netD(interpolates, real_labels)

    gradients = torch.autograd.grad(outputs=disc_interpolates,
                                    inputs=[interpolates, real_labels],
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
def calc_consistency_penalty(netD, real_data, real_labels, M):

    d1, d_1 = netD(real_data, real_labels)
    d2, d_2 = netD(real_data, real_labels)

    consistency_term = (d1 - d2).norm(2, dim=0) + 0.1 * (d_1 - d_2).norm(2, dim=1) - M
    return consistency_term.mean()

def get_rnd_label_generator(CONFIG):
    batch_size = CONFIG.BATCH_SIZE.GAN.TRAIN

    if CONFIG.SUPERVISE_TYPE == 'binary':
        return lambda: (torch.rand(batch_size) * 2).type(torch.int).type(torch.float)

    elif CONFIG.SUPERVISE_TYPE == 'regress':
        return lambda: (torch.rand(batch_size) * 2 - 1).type(torch.float)


def generate_image_cond(netG, CONFIG):
    num_images = 6
    noise_dims = CONFIG.ARCHITECTURE.GAN.NOISE_DIMS
    h = CONFIG.MATR_SIZE
    netG.eval()
    noise = torch.randn(num_images, noise_dims).cuda().requires_grad_(False)
    if CONFIG.SUPERVISE_TYPE == 'binary':
        labels = torch.Tensor([[0], [0], [0],
                               [1], [1], [1]]).cuda().requires_grad_(False)
        samples = netG(noise, labels)
    elif CONFIG.SUPERVISE_TYPE == 'regress':
        age_m = make_tuple(CONFIG.AGE_INTERVAL)[0]
        age_M = make_tuple(CONFIG.AGE_INTERVAL)[1]
        lab1 = (age_m + torch.rand(num_images // 2) * (55 - age_m)).type(torch.int)
        lab2 = (55 + torch.rand(num_images // 2) * (age_M - 55)).type(torch.int)
        labels = torch.cat([lab1, lab2], 0).type(torch.float).cuda().requires_grad_(False)
        samples = netG(noise, (labels - (age_M - age_m) / 2) / (age_M - age_m))
    samples = samples.view(num_images, 1, h, h)
    netG.train()
    # print(samples.shape)
    return samples, labels


def save_generated_images(samples, labels, CONFIG, it):
    i = str(it)
    os.makedirs(CONFIG.LOG_DIR.GAN, exist_ok=True)
    filename = os.path.join(CONFIG.LOG_DIR.GAN, 'gen_img_it_' + i + '.png')
    b, chs, h, w = samples.shape
    imgs = samples.view(b, h, w).detach().cpu().data.numpy()
    labels = labels.view(b).detach().cpu().data.numpy()
    fig = plt.figure()
    for i in range(b):
        plt.subplot(2, 3, i + 1)
        plt.imshow(imgs[i], cmap='jet', interpolation='nearest', vmin=-1, vmax=1)
        plt.title('Age: ' + str(labels[i]))
        plt.axis('off')
    # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    # cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    # plt.colorbar(cax=cax)
    plt.savefig(filename)


# def make_img_grid(imgs, num_col=5, border=5):
#     b, chs, h, w = imgs.shape
#     imgs = imgs.view(b, h, w).detach().cpu().data.numpy()
#     imgs_top = imgs[:num_col]
#     if b == num_col * 2:
#         imgs_bot = imgs[num_col:]
#         num_row = 2
#     else:
#         raise ValueError
#
#     empty_grid = np.zeros(((num_row + 1) * border + num_row * h, (num_col + 1) * border + num_col * w))
#     empty_grid[:] = np.nan
#     # top
#     for i in range(num_col):
#         empty_grid[border:border + h, border + (border + w) * i:border + (border + w) * i + w] = imgs_top[i]
#     # bot
#     if b == num_col * 2:
#         for i in range(num_col):
#             empty_grid[2 * border + h:2 * (border + h), border + (border + w) * i:border + (border + w) * i + w] = \
#                 imgs_bot[i]
#
#     return empty_grid


def generate_training_images(netG, CONFIG, device):
    num_images = CONFIG.GEN_IMG_NUM

    netG.eval()
    h = CONFIG.MATR_SIZE
    b = CONFIG.BATCH_SIZE.GAN.GENERATE
    noise_dims = CONFIG.ARCHITECTURE.GAN.NOISE_DIMS
    if CONFIG.SUPERVISE_TYPE == 'binary':
        labels = torch.Tensor([[0], [1]]).repeat(
            num_images // 2, 1).cuda().detach()
        normed_labels = labels
    elif CONFIG.SUPERVISE_TYPE == 'regress':
        age_m = make_tuple(CONFIG.AGE_INTERVAL)[0]
        age_M = make_tuple(CONFIG.AGE_INTERVAL)[1]
        normed_labels = (torch.rand(num_images) * 2 - 1).cuda()
        labels = (age_m + (normed_labels+1) / 2 * (age_M - age_m)).type(torch.int)
    # print('labels_shape: ', labels.shape)
    images = np.empty((num_images, 1, h, h)).astype(np.float32)
    # for i in range(num_images // b):
    for i in tqdm(
            range(0, num_images // b),
            total=num_images // b,
            leave=False,
            dynamic_ncols=True,
    ):
        noise = torch.randn(b, noise_dims).cuda().requires_grad_(False)
        samples = netG(noise, normed_labels[i * b:(i + 1) * b])
        samples = samples.view(b, 1, h, h).detach().cpu().data.numpy().astype(np.float32)
        images[i * b:(i + 1) * b] = samples
    labels = labels.cpu().numpy().reshape(-1)
    netG.train()
    return (images, labels)
