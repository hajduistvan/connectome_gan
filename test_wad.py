import torch
from data_handling.dataset import UKBioBankDataset
from gan_metrics.calc_metrics import MetricCalculator
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from noised_learning import NoisedTrainer


def test_eval_fid(model_id, gpu_id, batch_size, aug_fn, epochs, log_dir, alpha, save_dir, dataset_file):
    noise_trainer = NoisedTrainer(gpu_id, save_dir, dataset_file,
                                  cfg=args.cfg)

    os.makedirs(log_dir, exist_ok=True)
    dataset2 = UKBioBankDataset(dataset_file, None, 'train')
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size, shuffle=True)
    wad_list = []
    max_steps = len(loader2) // 2
    globstep = 0
    calcer = MetricCalculator(model_id, dataset2, batch_size, gpu_id, batch=None)
    for epoch in range(epochs):
        print('ep ', epoch)
        iter_loader2 = iter(loader2)
        for step in range(max_steps):
            print('step ', step)
            inputs1 = next(iter_loader2)
            inputs2 = next(iter_loader2)
            with torch.no_grad():
                aug_inp = aug_fn(*inputs1)
                calcer.reset_ref_batch(inputs2)
                calcer.feed_batch(*aug_inp)
                if epoch == 0 and step == 0:
                    calcer.scatter_plot_activations(os.path.join(save_dir, str(alpha) + '.svg'))
                _, _, wad = calcer.calc_wad()
            if epoch == 0 and step == 0:
                noise_trainer.set_train_loader(aug_fn)
                testloss, testacc = noise_trainer.run_train_cycle(alpha)
            wad_list.append(wad)
        globstep += 1
    wad = np.array(wad_list)
    wad_mean = np.mean(wad)
    wad_std = np.std(wad)

    return wad_mean, wad_std, testloss, testacc


def get_diag_noise_fn_tan(alpha):
    def add_uni_noise(x, y):
        noise = 2 * torch.rand_like(x) - 1
        return (1 - alpha) * x + alpha * (noise + noise.transpose(-1, -2) / 2), y

    return add_uni_noise


def test_fid_monotonities(model_id, batch_size, epochs, gpu_id, save_dir, dataset_file):
    os.makedirs(save_dir, exist_ok=True)
    alphas = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    xlabel = 'Noise-original ratio'

    numpy_name = os.path.join(save_dir, "wad.npy")
    eps_name = os.path.join(save_dir, 'wad.eps')
    png_name = os.path.join(save_dir, 'wad.png')
    wad_means, wad_stds, testlosses, testaccs = [], [], [], []
    for alpha in alphas:
        aug_fn = get_diag_noise_fn_tan(alpha)
        wad_mean, wad_std, testloss, testacc = test_eval_fid(model_id, gpu_id, batch_size, aug_fn, epochs, save_dir,
                                                             alpha, dataset_file=dataset_file)
        wad_means.append(wad_mean)
        wad_stds.append(2 * wad_std)
        testlosses.append(testloss)
        testaccs.append(testacc)
    metric_numpy = np.array([wad_means, wad_stds, testlosses, testaccs])
    np.save(numpy_name, metric_numpy)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(alphas, wad_means, yerr=wad_stds)
    plt.title('WAD with respect to noise added to data')
    plt.xlabel(xlabel)
    plt.ylabel('Wasserstein Activation Distance')
    # plt.xscale('log')
    plt.yscale('log')
    fig.savefig(eps_name)
    fig.savefig(png_name)
    plt.close('all')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(alphas, testlosses, yerr=0)
    plt.title('Test loss with respect to noise added to data')
    plt.xlabel(xlabel)
    plt.ylabel('Test loss')
    # plt.xscale('log')
    # plt.yscale('log')
    fig.savefig(eps_name.replace('wad', 'loss'))
    fig.savefig(png_name.replace('wad', 'loss'))
    plt.close('all')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(alphas, testaccs, yerr=0)
    plt.title('Test acc with respect to noise added to data')
    plt.xlabel(xlabel)
    plt.ylabel('Test acc')
    # plt.xscale('log')
    # plt.yscale('log')

    fig.savefig(eps_name.replace('wad', 'acc'))
    fig.savefig(png_name.replace('wad', 'acc'))
    plt.close('all')


def scatter_plot_batch_activations(activations, labels, filename=None):
    activations = activations.view(activations.shape[0], -1).cpu().numpy()
    labels = labels.float().view(-1, 1).cpu().numpy()
    plt.scatter(activations[:, 0], activations[:, 1], c=labels.ravel())
    plt.savefig(filename)
    plt.close('all')


def plot_experiment(save_dir):
    alphas = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    xlabel = 'Noise-original ratio'
    numpy_name = os.path.join(save_dir, "wad.npy")
    eps_name = os.path.join(save_dir, 'wad.eps')
    png_name = os.path.join(save_dir, 'wad.png')

    wad_numpy = np.load(numpy_name)
    wad_means, wad_stds, testlosses, testaccs = wad_numpy[0], wad_numpy[1], wad_numpy[2], wad_numpy[3]
    figsize = (5, 4)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(alphas, wad_means, yerr=wad_stds)
    plt.grid(which='both', axis='both')
    plt.title('WAD with respect to noise added to data')
    plt.xlabel(xlabel)
    plt.ylabel('Wasserstein Activation Distance')
    # plt.xscale('log')
    plt.yscale('log')
    fig.savefig(eps_name)
    fig.savefig(png_name)
    plt.close('all')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(alphas, testlosses, yerr=0)
    plt.grid(which='both', axis='both')
    plt.title('Learning Loss with respect to noise added to data')
    plt.xlabel(xlabel)
    plt.ylabel('Learning Loss')
    # plt.xscale('log')
    # plt.yscale('log')
    fig.savefig(eps_name.replace('wad', 'loss'))
    fig.savefig(png_name.replace('wad', 'loss'))
    plt.close('all')


if __name__ == '__main__':
    model_id = 887
    batch_size = 7000
    epochs = 50
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='debug')
    parser.add_argument("--save_dir", type=str, default=os.path.join(os.getcwd(), 'experiment_results'))
    parser.add_argument("--cfg", type=str, default=os.path.join(os.getcwd(), 'config/learning_loss.yaml'))
    parser.add_argument("--dataset_file", type=str, help='Path to the dataset .npz file',
                        default=os.path.join(os.getcwd(), 'partitioned_dataset_gender.npz'))
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--load", type=str, default='1')
    args = parser.parse_args()
    if args.load == '0':
        test_fid_monotonities(model_id, batch_size, epochs, args.gpu_id,
                              save_dir=os.path.join(args.save_dir, args.exp_name), dataset_file=args.dataset_file)
    else:
        plot_experiment(save_dir=os.path.join(args.save_dir, args.exp_name))
