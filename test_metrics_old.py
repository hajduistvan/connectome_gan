import torch
from gan_metrics.calc_metrics import calc_stats, calculate_frechet_distance, calc_bin_diffs, count_bins
from data_handling.dataset import UKBioBankDataset
from gan_metrics.select_cnn import get_model
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import torch.nn.functional as F

# metrics = [calculate_frechet_distance]
# dataset_root = '/home/orthopred/repositories/Datasets/UK_Biobank'
# manually_selected_indeces = [156, 1187, 606]  # middle point, small, large
# num_epochs = 1
# val_dataset = UKBioBankDataset(dataset_root, None, 'val')
# gpu_id = 0
"""
Test 1: Consistency
We test that the distance metrics are the same between exclusive subsets of the real data.
"""


def get_corrects_noreduction(sigmoids, labels):
    threshold = 0.5
    """
    Operates on detached cuda tensors!
    For binary classification, and keeping the dimensionalities.
    :param outputs:
    :param labels:
    :return:
    """
    preds = (sigmoids > threshold).long()
    correct = (preds == labels.long())
    return correct


# Test 1.a: consistency of exclusive subsets of training data
# Test 1.a.1: FID distance

def test_eval_fid(model_id, gpu_id, batch_size, aug_fn, epochs, log_dir, aug_str, alpha):
    dataset_root = '/home/orthopred/repositories/Datasets/UK_Biobank'
    os.makedirs(log_dir, exist_ok=True)
    dataset2 = UKBioBankDataset(dataset_root, None, 'train')
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size, shuffle=True)
    model = get_model(gpu_id, concat_list_idx=model_id)
    fid_list = []
    bindiff_list = []
    max_steps = len(loader2) // 2
    globstep = 0
    # print('alpha: ', alpha)
    # print('max_step: ', max_steps)
    # print('epochs: ', epochs)
    for epoch in range(epochs):
        print('ep ', epoch)
        iter_loader2 = iter(loader2)
        for step in range(max_steps):
            print('step ', step)
            inputs1 = next(iter_loader2)
            inputs2 = next(iter_loader2)
            with torch.no_grad():
                aug_inp = aug_fn(*inputs1)
                _, conv2_1, _ = model(aug_inp[0].cuda(gpu_id))
                # print('alpha: ', alpha)
                # print('uniq: ', np.unique(conv2_1.cpu().numpy()[:,0]))
                _, conv2_2, _ = model(inputs2[0].cuda(gpu_id))
                # scatter_plot_batch_activations(conv2_1, aug_inp[1], filename=os.path.join(log_dir,
                #                                                                           'aug_' + aug_str + str(
                #                                                                               alpha) + '_' + str(
                #                                                                               globstep) + '.png'))
                # scatter_plot_batch_activations(conv2_2, inputs2[1], filename=os.path.join(log_dir,
                #                                                                           'orig_' + aug_str + str(
                #                                                                               alpha) + '_' + str(
                #                                                                               globstep) + '.png'))

                hist1 = count_bins(conv2_1)
                hist2 = count_bins(conv2_2)
                bin_diff = calc_bin_diffs(hist1, hist2)
                # mu1, sigma1 = calc_stats(conv2_1)
                # mu2, sigma2 = calc_stats(conv2_2)

                mu1_c0, sigma1_c0 = calc_stats(conv2_2[inputs1[1].view(-1) == 0])
                mu1_c1, sigma1_c1 = calc_stats(conv2_2[inputs1[1].view(-1) == 1])
                mu2_c0, sigma2_c0 = calc_stats(conv2_2[inputs1[1].view(-1) == 0])
                mu2_c1, sigma2_c1 = calc_stats(conv2_2[inputs1[1].view(-1) == 1])
                fid_c0 = calculate_frechet_distance(mu1_c0, mu2_c0, sigma1_c0, sigma2_c0)
                fid_c1 = calculate_frechet_distance(mu1_c1, mu2_c1, sigma1_c1, sigma2_c1)
                # return fid_c0, fid_c1, (fid_c0 + fid_c1) / 2
                # fid_score = calculate_frechet_distance(mu1, mu2, sigma1, sigma2)
                fid_score = (fid_c0 + fid_c1) / 2
            bindiff_list.append(bin_diff)
            fid_list.append(fid_score)
        globstep += 1
    fid = np.array(fid_list)
    fid_mean = np.mean(fid)
    fid_std = np.std(fid)

    bindiff = np.array(bindiff_list)
    bin_mean = np.mean(bindiff)
    bin_std = np.std(bindiff)
    return fid_mean, fid_std, bin_mean, bin_std


def test_fid(part1, model_id, batch_size, augmentation, num_epochs=1, plot=False):
    """

    :param part1: 'train' or 'val'
    :param model_id: model id from cnn_search's csv_listm e.g. 156
    :param batch_sizes: for gt and gen batch
    :param num_epochs: Number of epochs to iterate over
    :return:
    """
    exp_folder = '/home/orthopred/repositories/conn_gan/experiment_results/experiment_results_0405'
    dataset_root = '/home/orthopred/repositories/Datasets/UK_Biobank'
    gpu_id = 0
    dataset2 = UKBioBankDataset(dataset_root, None, 'train')
    loader2 = torch.utils.data.DataLoader(dataset2, batch_size, shuffle=True)

    if part1 == 'train':

        model = get_model(0, concat_list_idx=model_id)
        fid_list = []
        max_steps = len(loader2) // 2
        print(max_steps)
        for epoch in range(num_epochs):
            iter_loader2 = iter(loader2)
            for step in range(max_steps):
                print(step)
                fid = []
                inputs1 = next(iter_loader2)
                inputs2 = next(iter_loader2)

                with torch.no_grad():

                    conv1_1, conv2_1, out1 = model(augmentation(inputs1[0]).cuda(gpu_id))
                    conv1_2, conv2_2, out2 = model(inputs2[0].cuda(gpu_id))
                    sigmoid1 = torch.sigmoid(out1)
                    sigmoid2 = torch.sigmoid(out2)
                    loss1 = F.binary_cross_entropy_with_logits(out1, inputs1[1].cuda(gpu_id), reduction='none')
                    loss2 = F.binary_cross_entropy_with_logits(out2, inputs2[1].cuda(gpu_id), reduction='none')
                    correct_preds1 = get_corrects_noreduction(sigmoid1, inputs1[1].cuda(gpu_id))
                    correct_preds2 = get_corrects_noreduction(sigmoid2, inputs2[1].cuda(gpu_id))
                    # activations1 = [inputs1[0], conv1_1, conv2_1, out1, sigmoid1, loss1, correct_preds1]
                    # activations2 = [inputs2[0], conv1_2, conv2_2, out2, sigmoid2, loss2, correct_preds2]
                    activations1 = [conv1_1, conv2_1, out1, sigmoid1, loss1, correct_preds1]
                    activations2 = [conv1_2, conv2_2, out2, sigmoid2, loss2, correct_preds2]
                for act1, act2 in zip(activations1, activations2):
                    mu1, sigma1 = calc_stats(act1)
                    mu2, sigma2 = calc_stats(act2)
                    fid_score = calculate_frechet_distance(mu1, mu2, sigma1, sigma2)
                    fid.append(fid_score)
                fid_list.append(fid)
                # print(fid)
    else:

        dataset1 = UKBioBankDataset(dataset_root, None, part1)
        loader1 = torch.utils.data.DataLoader(dataset1, batch_size, shuffle=True)
        iter_loader1 = iter(loader1)
        model = get_model(0, concat_list_idx=model_id)
        fid_list = []
        max_steps = len(loader2) - 1
        print(max_steps)
        for epoch in range(num_epochs):
            iter_loader2 = iter(loader2)
            for step in range(max_steps):
                print(step)
                fid = []
                inputs2 = next(iter_loader2)
                try:
                    inputs1 = next(iter_loader1)
                except:
                    iter_loader1 = iter(loader1)
                    inputs1 = next(iter_loader1)
                if inputs1[0].shape[0] != inputs2[0].shape[0]:
                    iter_loader1 = iter(loader1)
                    inputs1 = next(iter_loader1)
                with torch.no_grad():

                    conv1_1, conv2_1, out1 = model(augmentation(inputs1[0]).cuda(gpu_id))
                    conv1_2, conv2_2, out2 = model(inputs2[0].cuda(gpu_id))
                    sigmoid1 = torch.sigmoid(out1)
                    sigmoid2 = torch.sigmoid(out2)
                    loss1 = F.binary_cross_entropy_with_logits(out1, inputs1[1].cuda(gpu_id), reduction='none')
                    loss2 = F.binary_cross_entropy_with_logits(out2, inputs2[1].cuda(gpu_id), reduction='none')
                    correct_preds1 = get_corrects_noreduction(sigmoid1, inputs1[1].cuda(gpu_id))
                    correct_preds2 = get_corrects_noreduction(sigmoid2, inputs2[1].cuda(gpu_id))
                    # activations1 = [inputs1[0], conv1_1, conv2_1, out1, sigmoid1, loss1, correct_preds1]
                    # activations2 = [inputs2[0], conv1_2, conv2_2, out2, sigmoid2, loss2, correct_preds2]
                    activations1 = [conv1_1, conv2_1, out1, sigmoid1, loss1, correct_preds1]
                    activations2 = [conv1_2, conv2_2, out2, sigmoid2, loss2, correct_preds2]
                for act1, act2 in zip(activations1, activations2):
                    mu1, sigma1 = calc_stats(act1)
                    mu2, sigma2 = calc_stats(act2)
                    fid_score = calculate_frechet_distance(mu1, mu2, sigma1, sigma2)
                    fid.append(fid_score)
                fid_list.append(fid)

    fid_array = np.array(fid_list)
    # print(fid_array.shape)
    if False:
        np.save(os.path.join(exp_folder, 'fid_arr_cons_' + part1 + '_' + str(batch_size) + '.npy'),
                fid_array)

    fid_means = np.mean(fid_array, 0)
    fid_std = np.std(fid_array, 0)
    fid_rel_std = fid_std / fid_means
    print('FID means: ', fid_means, 'FID stds: ', fid_std, 'FID relative std: ', fid_rel_std)
    if plot:
        rng = np.arange(fid_array.shape[0])

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(len(fid_array[0, :])):
            ax.plot(rng, fid_array[:, i], label='Layer ' + str(i + 1))
        # Get artists and labels for legend and chose which ones to display
        handles, labels = ax.get_legend_handles_labels()
        display = (0, 3)
        ax.legend([handle for i, handle in enumerate(handles) if i in display],
                  [label for i, label in enumerate(labels) if i in display], loc='best')
        fig.savefig(os.path.join(exp_folder, 'fid_arr_cons_' + part1 + '_' + str(batch_size) + '.png'))
        plt.show()
    return fid_means, fid_std


def get_identity_fn():
    return lambda x, y: (x, y)


def get_diag_noise_fn_tan(alpha):
    def add_uni_noise(x, y):
        noise = torch.rand_like(x)
        return (1 - alpha) * x + alpha * (noise + noise.transpose(-1, -2) / 2), y

    return add_uni_noise


def get_saltpepper_freq(prob):
    def random_saltpepper(x, y):
        mask1 = torch.rand(*x.shape).le(prob)
        mask2 = torch.rand(*x.shape).le(0.5) * mask1
        return x.masked_fill_(mask1, 1).masked_fill(mask2, -1), y

    return random_saltpepper


def get_crossmagnitude_alpha(alpha):
    n = 3

    def random_cross_alpha(x, y):
        indeces = [np.random.permutation(x.shape[-1])[:n] for _ in range(x.shape[0])]
        mask_l = []
        for b in range(x.shape[0]):
            m = torch.ones(1, 1, 55, 55, dtype=x.dtype)
            m[:, :, :, indeces[b]] = float(alpha)
            m[:, :, indeces[b], :] = float(alpha)
            m[:, :, indeces[b], indeces[b]] = 1
            mask_l.append(m)
        mask = torch.cat(mask_l, 0)

        # multiplier = torch.ones_like(x)
        # multiplier[:,:,:,indeces] = alpha
        # multiplier[:,:,indeces,:] = alpha
        # multiplier = 1/2*(multiplier+multiplier.transpose(-1,-2))
        x = torch.atan(torch.tan(x * np.pi / 2) * mask) * 2 / np.pi
        return x, y

    return random_cross_alpha


def get_crossmagnitude_n(n):
    alpha = -1  # 1500

    def random_cross_n(x, y):
        indeces = [np.random.permutation(x.shape[-1])[:n] for _ in range(x.shape[0])]
        mask_l = []
        for b in range(x.shape[0]):
            m = torch.ones(1, 1, 55, 55, dtype=x.dtype)
            m[:, :, :, indeces[b]] = alpha
            m[:, :, indeces[b], :] = alpha
            m[:, :, indeces[b], indeces[b]] = 1
            mask_l.append(m)
        mask = torch.cat(mask_l, 0)
        # x = torch.atan(torch.tan(x * np.pi / 2) * mask) * 2 / np.pi
        x = x * mask
        return x, y

    return random_cross_n


def get_collapse_modes(n_modes):
    # n_modes: how many modes should be kept.
    def collapse_modes(x, y):
        indeces = np.arange(n_modes + 1) * x.shape[0] // n_modes
        # print('indeces: ', indeces)

        selected = [np.random.randint(indeces[i], indeces[i + 1]) for i in range(n_modes)]
        # print('selected: ', selected)
        for i in range(n_modes):
            # print(x[indeces[i]:indeces[i + 1], 0, 5, 6])
            # print(x[selected[i], 0, 5, 6])
            x[indeces[i]:indeces[i + 1], :, :, :] = x[selected[i], :, :, :]
            y[indeces[i]:indeces[i + 1]] = y[selected[i]]
            # print('unique: ', n_modes, np.unique(x.cpu().numpy()[:, 0, 5, 6]))
        return x, y

    return collapse_modes


def get_getter_and_alphas(aug_str):
    alphas = {'id': 1,
              'uniform': np.array([0, 0.0001, 0.001, 0.01, 0.1, 0.5, 1]),
              'saltpepper_freq': np.arange(6) / 10,
              'crossmagnitude_alpha': np.array([1, 1.1, 1.5, 2, 5, 10]) * 1000,
              'crossmagnitude_n': np.arange(6),
              'collapse_modes': np.array([1, 2, 3, 4, 5, 10, 50, 100, 500, 1000, 5000, 7000])}[aug_str]
    getter_fn = {'id': get_identity_fn,
                 'uniform': get_diag_noise_fn_tan,
                 'saltpepper_freq': get_saltpepper_freq,
                 'crossmagnitude_alpha': get_crossmagnitude_alpha,
                 'crossmagnitude_n': get_crossmagnitude_n,
                 'collapse_modes': get_collapse_modes}[aug_str]
    xlabel = {'id': 'Noise-original ratio',
              'uniform': 'Noise-original ratio',
              'saltpepper_freq': 'Noise probability',
              'crossmagnitude_alpha': 'Magnification of contaminated rows/coloumns',
              'crossmagnitude_n': 'Size of contamianted rows/coloumns',
              'collapse_modes': 'Number of modes kept'}[aug_str]
    return getter_fn, alphas, xlabel


def test_fid_monotonities(model_id, batch_size, aug_str, epochs, gpu_id, save_dir):
    save_dir = os.path.join('/home/orthopred/repositories/conn_gan/experiment_results', save_dir)
    os.makedirs(save_dir, exist_ok=True)
    aug_fn_getter, alphas, xlabel = get_getter_and_alphas(aug_str)

    numpy_name = os.path.join(save_dir, aug_str + "_full.npy")
    eps_name = os.path.join(save_dir, aug_str)
    fid_means, fid_stds, bin_means, bin_stds = [], [], [], []
    for alpha in alphas:
        aug_fn = aug_fn_getter(alpha)
        fid_mean, fid_std, bin_mean, bin_std = test_eval_fid(model_id, gpu_id, batch_size, aug_fn, epochs, save_dir, aug_str, alpha)
        fid_means.append(fid_mean)
        fid_stds.append(2*fid_std)
        bin_means.append(bin_mean)
        bin_stds.append(bin_std)
    metric_numpy = np.array([fid_means, fid_stds, bin_means, bin_stds])
    np.save(numpy_name, metric_numpy)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(alphas, fid_means, yerr=fid_stds)
    plt.title('WAD with respect to noise added to data')
    plt.xlabel(xlabel)
    plt.ylabel('Wasserstein Activation DIstance')
    plt.xscale('log')
    plt.yscale('log')

    fig.savefig(eps_name+ "fid_full.eps")
    plt.close('all')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(alphas, bin_means, yerr=bin_stds)
    plt.xlabel(xlabel)
    plt.ylabel('Bin Diff Score')
    # plt.xscale('log')
    # plt.yscale('log')

    fig.savefig(eps_name+ "bin_full.eps")
    plt.close('all')

    # plt.show()


def scatter_plot_batch_activations(activations, labels, filename=None):
    activations = activations.view(activations.shape[0], -1).cpu().numpy()
    labels = labels.float().view(-1, 1).cpu().numpy()
    plt.scatter(activations[:, 0], activations[:, 1], c=labels.ravel())
    plt.savefig(filename)
    plt.close('all')


def plot_experiment(exp_name, aug_str, eps_out):
    save_dir = os.path.join('/home/orthopred/repositories/conn_gan/experiment_results', exp_name)

    aug_fn_getter, alphas, xlabel = get_getter_and_alphas(aug_str)

    numpy_name = os.path.join(save_dir, aug_str + "_full.npy")
    eps_name = os.path.join(save_dir, aug_str + eps_out)
    fid_numpy = np.load(numpy_name)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(alphas, fid_numpy[0], yerr=fid_numpy[1])
    plt.title('WAD with respect to noise added to data')
    plt.xlabel(xlabel)
    plt.ylabel('Wasserstein Activation DIstance')
    plt.xscale('log')
    # plt.yscale('log')

    fig.savefig(eps_name + "fid_full.eps")

    plt.close('all')

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.errorbar(alphas, fid_numpy[2], yerr=fid_numpy[3])
    # plt.xlabel(xlabel)
    # plt.ylabel('Bin distance Score')
    # plt.xscale('log')
    # plt.yscale('log')
    #
    # fig.savefig(eps_name + "bin_full.eps")
    # plt.close('all')

if __name__ == '__main__':
    model_id = 1187
    batch_size = 7000
    epochs = 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='uni_0528')
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--aug", type=str, default='uniform')
    parser.add_argument("--load", type=str, default='full')
    args = parser.parse_args()
    if args.load == '0':
        test_fid_monotonities(model_id, batch_size, args.aug, epochs, args.gpu_id, args.exp_name)
    else:
        plot_experiment(args.exp_name, args.aug, args.load)
