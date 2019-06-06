import argparse
import torch
import yaml
from addict import Dict
from data_handling.dataset import UKBioBankDataset
import time
import os
import csv
import numpy as np
from shutil import copyfile
from models.classifier import ConnectomeConvNet
from models.cond_gan import CondGAN
from models.dual_gan import DualGAN
from decimal import Decimal





def convert_to_float(df):
    for k, v in df.items():
        if type(v) == str:
            try:
                e = float(v)
            except:
                e = v
            df[k] = e

    return df



def main(args):


    runs_path = '/home/orthopred/repositories/conn_gan/gan_manual_search/runs'
    config_dir_path = '/home/orthopred/repositories/conn_gan/config'
    log_dir = os.path.join(runs_path, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    config_file = os.path.join(config_dir_path, args.config)
    config = convert_to_float(yaml.load(open(config_file)))
    copyfile(config_file, os.path.join(log_dir, 'config.yaml'))

    np.random.seed(config['man_seed'])
    torch.manual_seed(config['man_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    model_cls, model_py = {
        'condgan': (CondGAN, '/home/orthopred/repositories/conn_gan/models/cond_gan.py'),
        'dualgan': (DualGAN, '/home/orthopred/repositories/conn_gan/models/dual_gan.py')
    }[config['gan_name']]
    copyfile(model_py, os.path.join(log_dir, 'gan_model.py'))



    # Datasets & Loaders
    train_dataset = UKBioBankDataset(args.dataset_root, None, 'train')
    val_dataset = UKBioBankDataset(args.dataset_root, None, 'val')

    val_loader = torch.utils.data.DataLoader(val_dataset, config['batch_size'], shuffle=False,
                                             num_workers=config['num_workers'])

    start_time = time.time()
    print("\n########################################################")
    model = model_cls(
        config,
        train_dataset,
        val_loader,
        args.gpu_id,
        config['fid_interval'],
        '0',
        log_dir,
        config['num_epochs'],
    )

    model.run_train()
    print("\nTraining took ", str(np.around(time.time() - start_time, 2)) + " seconds.")
    print("########################################################\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='debug')
    parser.add_argument("--config", type=str, default='gan_cfg.yaml')
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--num_epochs", type=int, default=100)
    args = parser.parse_args()

    main(args)
