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
from decimal import Decimal


def get_range_from_item_dict(item_dict):
    minval = float(item_dict['min'])
    maxval = float(item_dict['max'])
    step = float(item_dict['step'])
    if item_dict['log']:
        num_steps = int(np.around((np.log(maxval / minval) / np.log(step)))) + 1
        rng = np.power(step, np.arange(num_steps)) * minval
    else:
        num_steps = int(np.around((maxval - minval) / step + 1))
        rng = minval + step * np.arange(num_steps)
    return rng


def get_hyperparameter_range_from_cfg(config, gpu_id, split_id):
    ranges_dict = {}
    max_split_per_gpu = 3
    for item in config.items():
        if type(item[1]) == Dict:
            rng = get_range_from_item_dict(item[1])
            if item[0] in ['b1_g', 'b2_g', 'b1_d', 'b2_d']:
                rng = 1 - rng
            if item[0] == 'base_dim':
                if gpu_id == 0:
                    rng = rng[:len(rng) // 2]
                else:
                    rng = rng[len(rng) // 2:]
            if item[0] == 'noise_dim':
                if split_id == 0:
                    rng = rng[:len(rng) // 3]
                elif split_id == 1:
                    rng = rng[len(rng) // 3:len(rng) // 3 * 2]
                else:
                    rng = rng[len(rng) // 3 * 2]
            ranges_dict[item[0]] = rng
        else:
            ranges_dict[item[0]] = item[1]
    return ranges_dict


def rnd_sample_array(arr):
    return arr[np.random.randint(0, len(arr))]


def get_hyperparams_from_range_dict(rng_dict, debug_mode):
    hyperparams = {}
    hyperparams['p3'] = int(rnd_sample_array(rng_dict['base_dim']))
    hyperparams['p2'] = int(hyperparams['p3'] * rnd_sample_array(rng_dict['channel_multipliers']))
    hyperparams['p1'] = int(hyperparams['p3'] * rnd_sample_array(rng_dict['channel_multipliers']))
    hyperparams['q4'] = int(rnd_sample_array(rng_dict['base_dim']))
    hyperparams['q3'] = int(hyperparams['q4'] * rnd_sample_array(rng_dict['channel_multipliers']))
    hyperparams['q2'] = int(hyperparams['q3'] * rnd_sample_array(rng_dict['channel_multipliers']))
    hyperparams['q1'] = int(hyperparams['q3'] * rnd_sample_array(rng_dict['channel_multipliers']))

    for k, v in rng_dict.items():
        if type(v) == np.ndarray:
            if k in ['noise_dim', 'critic_iters']:
                hyperparams[k] = int(rnd_sample_array(v))
            else:
                hyperparams[k] = rnd_sample_array(v)
        else:
            hyperparams[k] = v
    if debug_mode:
        hyperparams['p3'] = 16
        hyperparams['p2'] = 32
        hyperparams['p1'] = 32
        hyperparams['q4'] = 8
        hyperparams['q3'] = 16
        hyperparams['q2'] = 32
        hyperparams['q1'] = 32
        hyperparams['noise_dim'] = 16
        hyperparams['b2_d'] = 0.99
        hyperparams['b2_g'] = 0.99
        hyperparams['lr_g'] = 1e-2
        hyperparams['lr_d'] = 1e-2

    return hyperparams


def main(args):
    runs_path = '/home/orthopred/repositories/conn_gan/gan_arch_search/runs'
    config_dir_path = '/home/orthopred/repositories/conn_gan/config'
    log_dir = os.path.join(runs_path, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    results_list = []
    start_step = 0
    if os.path.isfile(os.path.join(log_dir, str(args.gpu_id)+str(args.split_id)+"backup.pth")):
        config_file = os.path.join(log_dir, 'config.yaml')
        config = Dict(yaml.load(open(config_file)))
        load_dict = torch.load(os.path.join(log_dir, str(args.gpu_id)+str(args.split_id)+"backup.pth"))
        start_step = load_dict['step']
        results_list = load_dict['results_list']
        np.random.set_state(load_dict['numpy_rng'])
        torch.set_rng_state(load_dict['torch_rng'])
    else:
        config_file = os.path.join(config_dir_path, args.config)
        config = Dict(yaml.load(open(config_file)))
        copyfile(config_file, os.path.join(log_dir, 'config.yaml'))

    np.random.seed(config.man_seed)
    torch.manual_seed(config.man_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # hyperparam split

    ranges_dict = get_hyperparameter_range_from_cfg(config, args.gpu_id, args.split_id)

    val_interval = 1
    fid_interval = 10

    # Datasets & Loaders
    train_dataset = UKBioBankDataset(args.dataset_root, None, 'train')
    val_dataset = UKBioBankDataset(args.dataset_root, None, 'val')

    val_loader = torch.utils.data.DataLoader(val_dataset, config.batch_size, shuffle=False,
                                             num_workers=config.num_workers)

    for step in range(start_step, args.num_runs):
        hyperparameters = get_hyperparams_from_range_dict(ranges_dict, args.debug_mode)
        # print(hyperparameters)
        run_id = 'gpu' + str(args.gpu_id) + '_split' + str(args.split_id) + '_step' + str(step)
        start_time = time.time()
        print("\n########################################################")
        print("Starting step ", str(step) + " on GPU #" + str(args.gpu_id) + ".")
        model = CondGAN(
            hyperparameters,
            train_dataset,
            val_loader,
            args.gpu_id,
            fid_interval,
            run_id,
            log_dir,
            args.num_epochs,
        )

        fid, bindist = model.run_train()
        result = [args.gpu_id, args.split_id, step, fid, bindist, args.num_epochs, *hyperparameters.values()]
        results_list.append(result)
        print("\nTraining took ", str(np.around(time.time() - start_time, 2)) + " seconds.")
        print("FID Score: ", np.around(fid, 4))
        print("Bin Distance: ", np.around(bindist, 4))
        print("########################################################\n")
        save_dict = {
            'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'results_list': results_list,
            'step': step + 1
        }
        torch.save(save_dict, os.path.join(log_dir, str(args.gpu_id)+str(args.split_id)+"backup.pth"))
    # saving results to csv
    with open(os.path.join(log_dir, "cnn_search_result"+str(args.gpu_id)+str(args.split_id)+".csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(results_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='debug')
    parser.add_argument("--config", type=str, default='cond_gan_search.yaml')
    parser.add_argument("--debug_mode", type=bool, default=False, help='If True, trains for only one sensible model')
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=1000)
    parser.add_argument("--dataset_root", type=str)
    parser.add_argument("--num_epochs", type=int, default=100)
    # parser.add_argument("--load", type=bool)
    args = parser.parse_args()

    main(args)
