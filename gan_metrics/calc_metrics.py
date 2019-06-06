import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
from scipy import linalg
from gan_metrics.select_cnn import get_model
import torch
import matplotlib.pyplot as plt


def calculate_frechet_distance(mu1, mu2, sigma1, sigma2, eps=1e-6):
    """Pytorch implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Pytorch tensor containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calc_bin_diffs(hist1, hist2):
    return np.sum((hist1 - hist2) ** 2)


def count_bins(act):
    bins = 20
    rng = [[-5, 7], [-0.06, 0.02]]
    act = act.cpu().numpy()
    hist, xedges, yedges = np.histogram2d(act[:, 0], act[:, 1], bins=bins, range=rng)
    return hist


def calc_stats(act):
    """
    Calculates mean and covariance of activations.
    :param act: Torch cuda tensor of activations. First dim should be batch dimension.
    :return:
    """
    act = act.view(act.shape[0], -1).cpu().numpy()
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


class MetricCalculator:
    def __init__(
            self,
            model_id,
            real_dataset,
            batch_size,
            gpu_id,
            batch=None
    ):
        self.model_id = model_id
        self.real_dataset = real_dataset
        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.w = [-1.5350, -0.0037, 1.7972]
        self.inference_net = get_model(self.gpu_id, concat_list_idx=self.model_id)
        # print('inferencenet layer3 weights:')
        # print(self.inference_net.layer3.weight)
        # print(self.inference_net.layer3.bias)
        if batch is None:
            loader = torch.utils.data.DataLoader(self.real_dataset, batch_size, shuffle=False)
            iterloader = iter(loader)
            batch = next(iterloader)
        with torch.no_grad():
            self.labels = batch[1]
            self.batch = batch[0]
            _, self.r_act_2, self.r_act_3 = self.inference_net(self.batch.cuda(self.gpu_id))

            self.mu2, self.sigma2 = calc_stats(self.r_act_2)
            self.hist2 = count_bins(self.r_act_2)
            self.hist2_c0 = count_bins(self.r_act_2[self.labels.view(-1) == 0])
            self.hist2_c1 = count_bins(self.r_act_2[self.labels.view(-1) == 1])

            self.mu2_c0, self.sigma2_c0 = calc_stats(self.r_act_2[self.labels.view(-1) == 0])
            self.mu2_c1, self.sigma2_c1 = calc_stats(self.r_act_2[self.labels.view(-1) == 1])

    def reset_ref_batch(self, batch):
        with torch.no_grad():
            self.labels = batch[1]
            self.batch = batch[0]
            _, self.r_act_2, self.r_act_3 = self.inference_net(self.batch.cuda(self.gpu_id))

            self.mu2, self.sigma2 = calc_stats(self.r_act_2)
            # self.hist2 = count_bins(self.r_act_2)
            # self.hist2_c0 = count_bins(self.r_act_2[self.labels.view(-1) == 0])
            # self.hist2_c1 = count_bins(self.r_act_2[self.labels.view(-1) == 1])

            self.mu2_c0, self.sigma2_c0 = calc_stats(self.r_act_2[self.labels.view(-1) == 0])
            self.mu2_c1, self.sigma2_c1 = calc_stats(self.r_act_2[self.labels.view(-1) == 1])

    def feed_batch(self, generated_batch, generated_labels):
        _, self.act2, self.act3 = self.inference_net(generated_batch.cuda(self.gpu_id))
        self.g_labels = generated_labels

    def calc_fid(self):
        mu1, sigma1 = calc_stats(self.act2)
        fid_score = calculate_frechet_distance(mu1, self.mu2, sigma1, self.sigma2)
        return fid_score

    def calc_class_agnostic_fid(self):
        mu1_c0, sigma1_c0 = calc_stats(self.act2[self.g_labels.view(-1) == 0])
        mu1_c1, sigma1_c1 = calc_stats(self.act2[self.g_labels.view(-1) == 1])
        fid_c0 = calculate_frechet_distance(mu1_c0, self.mu2_c0, sigma1_c0, self.sigma2_c0)
        fid_c1 = calculate_frechet_distance(mu1_c1, self.mu2_c1, sigma1_c1, self.sigma2_c1)
        return fid_c0, fid_c1, (fid_c0+fid_c1)/2

    def calc_bin_class_agnostic(self):
        hist1_c0 = count_bins(self.act2[self.g_labels.view(-1) == 0])
        hist1_c1 = count_bins(self.act2[self.g_labels.view(-1) == 1])
        bin_dist_c0 = calc_bin_diffs(hist1_c0, self.hist2_c0)
        bin_dist_c1 = calc_bin_diffs(hist1_c1, self.hist2_c1)
        return bin_dist_c0, bin_dist_c1, (bin_dist_c0+bin_dist_c1)/2

    def calc_bin_dist(self):
        hist1 = count_bins(self.act2)
        bin_dist = calc_bin_diffs(hist1, self.hist2)
        return bin_dist

    def scatter_plot_activations(self, filename):
        y = np.linspace(-0.06, 0.02)
        x = -self.w[2] / self.w[0] - self.w[1] / self.w[0] * y
        plt.plot(x, y)
        r_act = self.r_act_2.view(self.r_act_2.shape[0], -1).cpu().numpy()
        r_labels = self.labels.float().view(-1).cpu().numpy()
        act = self.act2.view(self.act2.shape[0], -1).cpu().numpy()
        labels = self.g_labels.float().view(-1).cpu().numpy()
        l2 = plt.scatter(r_act[r_labels == 1, 0], r_act[r_labels == 1, 1], c='crimson', marker='.', alpha=0.7,
                         edgecolors='none')
        l1 = plt.scatter(r_act[r_labels == 0, 0], r_act[r_labels == 0, 1], c='gold', marker='.', alpha=0.3,
                         edgecolors='none')
        l4 = plt.scatter(act[labels == 1, 0], act[labels == 1, 1], c='navy', marker='.', alpha=0.7, edgecolors='none')
        l3 = plt.scatter(act[labels == 0, 0], act[labels == 0, 1], c='aqua', marker='.', alpha=0.3, edgecolors='none')

        plt.legend((l1, l2, l3, l4),
                   ('Real, Female', 'Real, Male', 'Gen, Female', 'Gen, Male'))
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.savefig(filename)
        plt.savefig(filename.replace('svg', 'png'))
        plt.close('all')
    def plot_empty_scatter(self, filename):
        y = np.linspace(-0.06, 0.02)
        x = -self.w[2] / self.w[0] - self.w[1] / self.w[0] * y
        plt.plot(x, y)
        r_act = self.r_act_2.view(self.r_act_2.shape[0], -1).cpu().numpy()
        r_labels = self.labels.float().view(-1).cpu().numpy()
        l2 = plt.scatter(r_act[r_labels == 1, 0], r_act[r_labels == 1, 1], c='crimson', marker='.', alpha=0.7,
                         edgecolors='none')
        l1 = plt.scatter(r_act[r_labels == 0, 0], r_act[r_labels == 0, 1], c='gold', marker='.', alpha=0.3,
                         edgecolors='none')

        plt.legend((l1, l2),
                   ('Real, Female', 'Real, Male'))
        plt.xlabel('x_1')
        plt.ylabel('x_2')
        plt.savefig(filename)
        plt.savefig(filename.replace('png', 'eps'))
        plt.savefig(filename.replace('png', 'svg'))

        plt.close('all')

    def calc_crossentropy(self):
        ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(self.act3.view(-1), self.g_labels.float())
        return ce_loss


if __name__=='__main__':
    metric_model_id = 1187
    batch_size = 7000
    gpu_id=0
    from data_handling.dataset import UKBioBankDataset
    dataset = UKBioBankDataset('/home/orthopred/repositories/Datasets/UK_Biobank')
    calcer = MetricCalculator(metric_model_id, dataset, batch_size, gpu_id)
    calcer.plot_empty_scatter(os.path.join('/home/orthopred/repositories/conn_gan/cnn_arch_search', 'empty_scatter.png'))
