import torch
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
    """
    Used for testing WAD. Trains the CNN defined by cfg with noise added to the whole training set. Used in test_wad.py.
    """
    def __init__(self, gpu_id, log_dir, dataset_root, cfg):
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
        """
        Uses aug_fn to add noise to the training dataset.
        :param aug_fn:
        :param num_total_examples:
        :return:
        """
        real_dataset = UKBioBankDataset(self.dataset_root, num_total_examples, 'train', aug_fn=aug_fn)
        self.train_loader = torch.utils.data.DataLoader(real_dataset, self.config_class.batch_size, shuffle=True,
                                                        num_workers=self.config_class.num_workers)

    def run_train_cycle(self, run_id):
        """
        Trains the classifier, then tests it.
        :param run_id:
        :return: the test loss and acc.
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
        _, _, _ = classifier.run_train()
        testloss, testacc = classifier.test(self.test_loader)
        print(testloss, testacc)
        return testloss, testacc
