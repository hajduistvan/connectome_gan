import numpy as np
import torch
import os
from torch.utils import data


class UKBioBankDataset(data.Dataset):
    """
    Torch Dataset for the UK Biobank conenctivity matrices dataset.
    """

    def __init__(
            self,
            dataset_file,
            number_of_examples=None,
            split='train',
            random_noise_hardening=False,
            aug_fn=None
    ):
        """
        Init.
        :param dataset_file: Path to the 'partitioned_dataset_gender.npz' file that contains the train, val, and test dataset.
        :param number_of_examples: number of dataset examples. set to None if all of the data is to be used.
        :param split: 'train', 'val', or 'test'.
        :param random_noise_hardening: randomly change matrices to random noise.
        :param aug_fn: Used in WAD testing, applies aug_fn to matrices. Usually adds uniuform noise.
        """
        self.random_noise_hardening = random_noise_hardening
        with np.load(dataset_file) as df:
            dataset = df[split + '_dataset']
        self.images = dataset[:number_of_examples, 0]
        self.labels = dataset[:number_of_examples, 1]
        self.aug_fn = aug_fn

    def __getitem__(self, index):
        """
        Returns matrix of id idx.
        :param index:
        :return:
        """
        image, label = torch.Tensor(self.images[index]), torch.Tensor([self.labels[index]])
        if not self.aug_fn is None:
            image, label = self.aug_fn(image, label)
        if self.random_noise_hardening and np.random.rand() < 0.5:
            image = torch.rand_like(image)
        return image.type(torch.float).view(1, 55, 55), label.type(torch.float).view(1)

    def __len__(self):
        return len(self.images)


class GenDataset(data.Dataset):
    """
    Used to construct Dataset from GAN-generated matrices.
    """

    def __init__(
            self,
            images,
            labels
    ):
        """
        :param images: the generated matrices
        :param labels: the labels of the amtrices
        """
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]

        return image.type(torch.float).view(1, 55, 55), label.type(torch.float).view(1)

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, help='Path to the dataset .npz file',
                        default=os.path.join(os.getcwd(), 'partitioned_dataset_gender.npz'))
    args = parser.parse_args()

    split = 'train'
    dataset = UKBioBankDataset(args.dataset_file, None, split)
    loader = torch.utils.data.DataLoader(dataset, 10, shuffle=False,
                                         num_workers=0)
    iterloader = iter(loader)
    for el in iterloader:
        im, lab = el
        print('image: ', np.unique(im), im.shape, im.dtype)
        print('label: ', np.unique(lab), lab.shape, lab.dtype)
