import argparse
import torch
import yaml
from data_handling.dataset import UKBioBankDataset
import time
import os
import numpy as np
from shutil import copyfile
from models.cond_gan import CondGAN


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

    # Simple training script.
    log_dir = os.path.join(args.runs_path, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    config = convert_to_float(yaml.load(open(args.config)))
    copyfile(args.config, os.path.join(log_dir, 'config.yaml'))

    np.random.seed(config['man_seed'])
    torch.manual_seed(config['man_seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_cls, model_py = {
        'condgan': (CondGAN, os.path.join(args.models_path, 'cond_gan.py')),
    }[config['gan_name']]
    copyfile(model_py, os.path.join(log_dir, 'gan_model.py'))

    # Datasets & Loaders
    train_dataset = UKBioBankDataset(args.dataset_file, None, 'train')
    val_dataset = UKBioBankDataset(args.dataset_file, None, 'val')

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
        log_dir,
        config['num_epochs'],
        args.cnn_run_dir,
        args.metric_model_id
    )

    model.run_train()
    print("\nTraining took ", str(np.around(time.time() - start_time, 2)) + " seconds.")
    print("########################################################\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, default='debug')
    parser.add_argument("--config", type=str, default=os.path.join(os.getcwd(), 'config/gan.yaml'))
    parser.add_argument("--runs_path", type=str, default=os.path.join(os.getcwd(), 'gan_runs'))
    parser.add_argument("--models_path", type=str, default=os.path.join(os.getcwd(), 'models'))
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--metric_model_id", type=int, default=0)
    parser.add_argument("--dataset_file", type=str, help='Path to the dataset .npz file',
                        default=os.path.join(os.getcwd(), 'partitioned_dataset_gender.npz'))
    parser.add_argument("--cnn_run_dir", type=str, default=os.path.join(os.getcwd(), 'cnn_arch_search/runs'))
    parser.add_argument("--num_epochs", type=int, default=10)
    args = parser.parse_args()

    main(args)
