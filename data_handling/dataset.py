import numpy as np
import torch
import os
from torch.utils import data


class UKBioBankDataset(data.Dataset):
    def __init__(
            self,
            dataset_root,
            number_of_examples=None,
            split='train',
            random_noise_hardening=False,
            aug_fn = None
    ):
        self.dataset_root = dataset_root
        self.random_noise_hardening = random_noise_hardening
        npz_filename = os.path.join(self.dataset_root, 'partitioned_dataset_gender.npz')
        with np.load(npz_filename) as df:
            dataset = df[split + '_dataset']
        self.images = dataset[:number_of_examples, 0]
        self.labels = dataset[:number_of_examples, 1]
        self.aug_fn = aug_fn
    def __getitem__(self, index):
        image, label = torch.Tensor(self.images[index]), torch.Tensor([self.labels[index]])
        if not self.aug_fn is None:
            image, label = self.aug_fn(image, label)
        if self.random_noise_hardening and np.random.rand() < 0.5:
            image = torch.rand_like(image)
        return image.type(torch.float).view(1, 55, 55), label.type(torch.float).view(1)

    def __len__(self):
        return len(self.images)


class GenDataset(data.Dataset):
    def __init__(
            self,
            images,
            labels
    ):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]

        return image.type(torch.float).view(1, 55, 55), label.type(torch.float).view(1)

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':

    root = '/home/orthopred/repositories/Datasets/UK_Biobank'
    split = 'train'
    dataset = UKBioBankDataset(root, None, split)
    loader = torch.utils.data.DataLoader(dataset, 10, shuffle=False,
                                         num_workers=0)
    iterloader = iter(loader)
    for el in iterloader:
        im, lab = el
        print('image: ', np.unique(im), im.shape, im.dtype)
        print('label: ', np.unique(lab), lab.shape, lab.dtype)
