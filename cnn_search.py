import argparse
import torch
import yaml
from addict import Dict
from data_handling.dataset import UKBioBankDataset
import time
import csv
import numpy as np
from shutil import copyfile
from models.classifier import ConnectomeConvNet
from decimal import Decimal


def get_log_range(min, max, log_step):
    """
    Calculates the points of interest for a hyperparameter.
    :param min: minimum value of the hyperparameter
    :param max: maximum value.
    :param log_step: multiplicatorial step, used for decreasing value from max to min.
    :return: the array of the hyperparameter range.
    """
    min, max, log_step = float(min), float(max), float(log_step)
    num_steps = int(np.around((np.log(max / min) / np.log(log_step)))) + 1
    rng = np.power(log_step, np.arange(num_steps)) * min
    return rng


def get_add_range(min, max, step):
    """
    Same as above, just with additive step.
    :param min:
    :param max:
    :param step:
    :return:
    """
    min, max, step = float(min), float(max), float(step)
    num_steps = int(np.around((max - min) / step + 1))
    rng = min + step * np.arange(num_steps)
    return rng


def main(args):
    log_dir = os.path.join(args.runs_path, args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    results_list = []
    start_step = 0
    if os.path.isfile(os.path.join(log_dir, "backup.pth")):

        # If we have already started the experiment, resume it
        config_file = os.path.join(log_dir, 'config.yaml')
        config = Dict(yaml.load(open(config_file)))
        load_dict = torch.load(os.path.join(log_dir, "backup.pth"))
        start_step = load_dict['step']
        results_list = load_dict['results_list']
        np.random.set_state(load_dict['numpy_rng'])
        torch.set_rng_state(load_dict['torch_rng'])
    else:
        config = Dict(yaml.load(open(args.config_file)))
        copyfile(args.config_file, os.path.join(log_dir, 'config.yaml'))
        np.random.seed(config.man_seed)
        torch.manual_seed(config.man_seed)

    # Deterministic training!
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # hyperparam split
    conv1_widths = get_log_range(*config.conv1.values()).astype(int)
    conv1_widths = conv1_widths[args.gpu_id * len(conv1_widths) // 2:
                                (args.gpu_id + 1) * len(conv1_widths) // 2]
    conv2_widths = get_log_range(*config.conv2.values()).astype(int)
    lr_values = get_log_range(*config.lr.values())
    mom_values = get_add_range(*config.mom.values())
    wd_values = get_log_range(*config.wd.values())

    val_interval = 1
    max_epochs = 200

    # Datasets & Loaders
    train_dataset = UKBioBankDataset(args.dataset_file, None, 'train')
    val_dataset = UKBioBankDataset(args.dataset_file, None, 'val')

    train_loader = torch.utils.data.DataLoader(train_dataset, config.batch_size, shuffle=True,
                                               num_workers=config.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, config.batch_size, shuffle=False,
                                             num_workers=config.num_workers)

    for step in range(start_step, args.num_runs):
        c1_w = conv1_widths[np.random.randint(0, len(conv1_widths))]
        c2_w = conv2_widths[np.random.randint(0, len(conv2_widths))]
        lr = lr_values[np.random.randint(0, len(lr_values))]
        mom = mom_values[np.random.randint(0, len(mom_values))]
        wd = wd_values[np.random.randint(0, len(wd_values))]
        start_time = time.time()
        print("\n########################################################")
        print("Starting step ", str(step) + " on GPU #" + str(args.gpu_id) + ".")
        print("conv1, conv2 widths: " + str((c1_w, c2_w)))
        print("lr: %.2E, mom: %.2E, wd_ %.2E\n" % (Decimal(lr), Decimal(mom), Decimal(wd)))
        model = ConnectomeConvNet(
            (c1_w, c2_w),
            lr,
            mom,
            wd,
            train_loader,
            val_loader,
            args.gpu_id,
            val_interval,
            step,
            log_dir,
            max_epochs,
        )
        loss, acc, num_params = model.run_train()
        result = [args.gpu_id, step, loss, acc, num_params, c1_w, c2_w, lr, mom, wd]
        results_list.append(result)
        print("\nTraining took ", str(np.around(time.time() - start_time, 2)) + " seconds.")
        print("Loss: ", np.around(loss, 4))
        print("Acc: ", np.around(acc, 4))
        print("Number of params: ", num_params)
        print("########################################################\n")
        save_dict = {
            'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'results_list': results_list,
            'step': step + 1
        }
        torch.save(save_dict, os.path.join(log_dir, "backup.pth"))
    # saving results to csv
    with open(os.path.join(log_dir, "cnn_search_result.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(results_list)


if __name__ == "__main__":
    # run the experiment!
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", type=str, default='debug')
    parser.add_argument("--runs_path", type=str,
                        help='Dir where checkpoints and tensorboard summaries are to be saved.',
                        default=os.path.join(os.getcwd(), 'cnn_arch_search/runs'))
    parser.add_argument("--config_file", type=str, help='Path to the search config file',
                        default=os.path.join(os.getcwd(), 'config/cnn_search.yaml'))
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_runs", type=int, default=1000)  # 1000
    parser.add_argument("--dataset_file", type=str, help='Path to the dataset .npz file',
                        default=os.path.join(os.getcwd(), 'partitioned_dataset_gender.npz'))
    args = parser.parse_args()

    main(args)
