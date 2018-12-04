"""
@ author István Hajdu at MTA TTK
https://github.com/hajduistvan/connectome_gan
"""
import numpy as np
import torch
from torch.utils import data
from ast import literal_eval as make_tuple
import os


# TODO: do we even need 2 separate loaders?

class Autism_handler(data.Dataset):
    def __init__(
            self,
            data_frame,
            supervise_type='binary',
            data_gen=None,
            split='train',
    ):
        self.data_frame = data_frame
        self.supervise_type = supervise_type
        self.data_gen = data_gen
        self.split = split
        self.images = data_frame[split][0]
        self.labels = data_frame[split][1]
        if not self.data_gen is None:
            self.images = np.concatenate((self.images, self.data_gen[0]))
            self.labels = np.concatenate((self.labels, self.data_gen[1]))

    def __getitem__(self, index):
        # TODO: is this working?
        # if index == len(self.data_frame[self.split][0])-1:
        #     index = 0
        image, label = self.images[index], self.labels[index]

        return image.astype(np.float32), label.astype(np.float32)

    def __len__(self):
        return len(self.data_frame[self.split][0])


class Age_handler(data.Dataset):
    def __init__(
            self,
            data_frame,
            config,
            split='train',
    ):
        self.data_frame = data_frame
        self.split = split
        self.images = data_frame[split][0]
        self.labels = data_frame[split][1]
        self.config = config

        self.images, self.labels = self._exclude_nans()
        self.images, self.labels = self.images[:self.images.shape[0] // 2 * 2], self.labels[
                                                                                :self.labels.shape[0] // 2 * 2]

        self.labels = self._norm_regress_labels()

    def _exclude_nans(self):
        batch_dim_mask = np.ones(self.images.shape[0])
        for i in range(self.images.shape[0]):
            batch_dim_mask[i] = np.sum(np.isnan(self.images[i])) == 0
        batch_dim_mask = batch_dim_mask.astype(np.bool)
        return self.images[batch_dim_mask, :, :, :], self.labels[batch_dim_mask]

    def _norm_regress_labels(self):
        # normalize to range -1,1
        age_m = make_tuple(self.config.AGE_INTERVAL)[0]
        age_M = make_tuple(self.config.AGE_INTERVAL)[1]
        labels = (self.labels - (age_M - age_m) / 2) / (age_M - age_m)
        return labels

    def __getitem__(self, index):
        image, label = torch.Tensor(self.images[index]), torch.Tensor([self.labels[index]])
        return image.type(torch.float), label.type(torch.float)

    def __len__(self):
        return len(self.images)


class UKBioBankHandler(data.Dataset):
    def __init__(
            self,
            config,
            number_of_examples = None,
            split='train',
            gen_values = None
    ):
        self.config = config
        if gen_values is None:
            npz_filename = os.path.join(config.DATA_FOLDER, config.DATASET_PART)
            with np.load(npz_filename) as df:
                dataset = df[split + '_dataset']
            self.images = dataset[:number_of_examples, 0]
            self.labels = dataset[:number_of_examples, 1]
        else:
            self.images, self.labels = gen_values



        self.labels = self._norm_regress_labels()

    def _norm_regress_labels(self):
        # normalize to range -1,1
        age_m = make_tuple(self.config.AGE_INTERVAL)[0]
        age_M = make_tuple(self.config.AGE_INTERVAL)[1]
        # labels = (self.labels - (age_M - age_m) / 2) / (age_M - age_m)
        labels = (self.labels - age_m - (age_M - age_m) / 2) / (age_M - age_m) * 2
        return labels

    def __getitem__(self, index):
        image, label = torch.Tensor(self.images[index]), torch.Tensor([self.labels[index]])
        return image.type(torch.float).view(1, self.config.MATR_SIZE, self.config.MATR_SIZE), label.type(
            torch.float).view(1)

    def __len__(self):
        return len(self.images)

class UKBioBankGenderHandler(data.Dataset):
    def __init__(
            self,
            config,
            number_of_examples=None,
            split='train',
            gen_values=None
    ):
        self.config = config
        if gen_values is None:
            npz_filename = os.path.join(config.DATA_FOLDER, config.DATASET_PART)
            with np.load(npz_filename) as df:
                dataset = df[split + '_dataset']
            self.images = dataset[:number_of_examples, 0]
            self.labels = dataset[:number_of_examples, 1]
        else:
            self.images, self.labels = gen_values


    def __getitem__(self, index):
        image, label = torch.Tensor(self.images[index]), torch.Tensor([self.labels[index]])
        return image.type(torch.float).view(1, self.config.MATR_SIZE, self.config.MATR_SIZE), label.type(
            torch.long).view(1)

    def __len__(self):
        return len(self.images)
