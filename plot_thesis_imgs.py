# -*- coding: utf-8 -*-
"""
@author István Hajdu at MTA TTK
https://github.com/hajduistvan/connectome_gan
"""
import torch
import numpy as np
import yaml
from data_handling.dataset import UKBioBankDataset, GenDataset
from models.cond_gan import CondGAN
from models.classifier import ConnectomeConvNet
import os
import matplotlib.pyplot as plt


def convert_to_float(df):
    for k, v in df.items():
        if type(v) == str:
            try:
                e = float(v)
            except:
                e = v
            df[k] = e

    return df


def plot_wad(ckpts_dir, save_dir, cfg_file, dataset_file):
    cfg = convert_to_float(yaml.load(open(cfg_file)))
    train_dataset = UKBioBankDataset(dataset_file, None, 'train')

    val_loader = torch.utils.data.DataLoader(train_dataset, cfg['batch_size'], shuffle=False,
                                             num_workers=cfg['num_workers'])
    model = CondGAN(cfg, train_dataset, val_loader, 0, 10, save_dir, 200, 887)

    # Datasets & Loaders

    os.makedirs(save_dir, exist_ok=True)
    fnames = ['ckpt_step_' + str(i) + '.pth' for i in range(0, 181)]
    steps = []
    wads = []
    png_name = os.path.join(save_dir, 'wad_plot.')

    for i, f in enumerate(fnames):
        print(i)
        fname = os.path.join(ckpts_dir, f)
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


def plot_wasserstein_loss(ckpts_dir, save_dir, cfg_file, dataset_file):
    cfg = convert_to_float(yaml.load(open(cfg_file)))
    train_dataset = UKBioBankDataset(dataset_file, None, 'train')

    val_loader = torch.utils.data.DataLoader(train_dataset, cfg['batch_size'], shuffle=False,
                                             num_workers=cfg['num_workers'])
    # Datasets & Loaders

    os.makedirs(save_dir, exist_ok=True)
    fname = 'ckpt_step_' + str(181) + '.pth'
    png_name = os.path.join(save_dir, 'wass_plot.')

    npy_log = torch.load(os.path.join(ckpts_dir, fname))['numpy_log']
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


def calc_learner_loss(ckpt_dir, save_dir, gan_cfg_file, learner_cfg_file, dataset_file, cnn_arch_dir):
    # steps = [0, 50, 100, 150, 174]
    # steps = [25, 75, 125, 180]
    # steps = [280, 320]
    steps = [254, 300]

    test_dataset = UKBioBankDataset(dataset_file, None, 'test')
    val_dataset = UKBioBankDataset(dataset_file, None, 'val')

    test_loader = torch.utils.data.DataLoader(test_dataset, learner_cfg_file.batch_size, shuffle=False,
                                              num_workers=learner_cfg_file.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, learner_cfg_file.batch_size, shuffle=False,
                                             num_workers=learner_cfg_file.num_workers)

    os.makedirs(save_dir, exist_ok=True)
    gan_model = CondGAN(
        convert_to_float(
            yaml.load(
                open(gan_cfg_file))),
        val_dataset,
        val_loader,
        1,
        1,
        ckpt_dir,
        200,
        cnn_arch_dir=cnn_arch_dir,
        metric_model_id=887
    )
    test_losses, test_accs = [], []

    for step in steps:
        fname = os.path.join(ckpt_dir, 'ckpt_step_' + str(step) + '.pth')
        gan_model.netg.load_state_dict(torch.load(fname)['netg'])
        imgs, labels = gan_model.netg.generate_fake_images(14861)
        gen_dataset = GenDataset(imgs.cpu(), labels.cpu())
        train_loader = torch.utils.data.DataLoader(gen_dataset, 14861, shuffle=True,
                                                   num_workers=0)

        classifier = ConnectomeConvNet(
            (learner_cfg_file.c1, learner_cfg_file.c2),
            learner_cfg_file.lr,
            learner_cfg_file.mom,
            learner_cfg_file.wd,
            train_loader,
            val_loader,
            1,
            1,
            0,
            os.path.join(save_dir, 'gen' + str(step)),
            200,
            allow_stop=False,
            verbose=True,
        )
        loss, acc, num_params = classifier.run_train()
        testloss, testacc = classifier.test(test_loader)
        print(step, testloss, testacc)
        test_losses.append(testloss)
        test_accs.append(test_accs)
    np.save(os.path.join(save_dir, 'loss_result.npy'), np.array([steps, test_losses, test_accs]))


def plot_generated_matrices(ckpt_dir, save_dir, dataset_file):
    os.makedirs(save_dir, exist_ok=True)
    fnames = ['gen_img_it_' + str(i) + '.npy' for i in range(0, 181)]

    # real example
    train_dataset = UKBioBankDataset(dataset_file, None, 'train')
    loader = torch.utils.data.DataLoader(train_dataset, 200, shuffle=True,
                                         num_workers=0)
    iterloader = iter(loader)
    batch = next(iterloader)
    female_examples = batch[0][(batch[1] == 0).view(-1), 0, :, :][0, :, :]
    male_examples = batch[0][(batch[1] == 1).view(-1), 0, :, :][0, :, :]
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
    plt.savefig(os.path.join(save_dir, 'real_example') + '.eps', bbox_inches='tight', pad_inches=0.0)
    plt.savefig(os.path.join(save_dir, 'real_example') + '.png', bbox_inches='tight', pad_inches=0.0)
    plt.close()

    # generated examples
    for j, fname in enumerate(fnames):
        fname = os.path.join(ckpt_dir, fname)
        img_fname = os.path.join(save_dir, 'gen_' + str(j))
        arr = np.load(fname)
        fig = plt.figure()
        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.imshow(arr[i, :, :], cmap='jet', interpolation='nearest', vmin=-1, vmax=1)
            plt.title('Sex: ' + ['Female', 'Male'][i])
            plt.axis('off')
        plt.savefig(img_fname + '.eps', bbox_inches='tight', pad_inches=0.0)
        plt.savefig(img_fname + '.png', bbox_inches='tight', pad_inches=0.0)
        plt.close()

    if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset_file", type=str, help='Path to the dataset .npz file',
                            default=os.path.join(os.getcwd(), 'partitioned_dataset_gender.npz'))
        parser.add_argument("--save_dir", type=str,
                            default=os.path.join(os.getcwd(), 'gan_runs/cond_gan_debug_24/plots'))
        parser.add_argument("--ckpts_dir", type=str,
                            default=os.path.join(os.getcwd(), 'gan_runs/cond_gan_debug_24/ckpts'))
        parser.add_argument("--cnn_runs_dir", type=str, default=os.path.join(os.getcwd(), 'cnn_arch_search'))

        parser.add_argument("--gan_cfg_file", type=str,
                            default=os.path.join(os.getcwd(), 'gan_runs/cond_gan_debug_24/config.yaml'))
        parser.add_argument("--learner_cfg_file", type=str,
                            default=os.path.join(os.getcwd(), 'config/learning_loss.yaml'))

        args = parser.parse_args()

        # Uncomment to use desired function!

        # plot_generated_matrices(args.ckpt_dir, args.save_dir, args.dataset_file)
        # plot_wasserstein_loss(args.ckpts_dir, args.save_dir, args.gan_cfg_file)
        # plot_wad(args.ckpts_dir, args.save_dir, args.gan_cfg_file)
        calc_learner_loss(args.ckpt_dir, args.save_dir, args.gan_cfg_file, args.learner_cfg_file, args.dataset_file,
                          args.cnn_arch_dir)
