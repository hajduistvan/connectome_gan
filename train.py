"""
@author István Hajdu at MTA TTK
https://github.com/hajduistvan/connectome_gan
"""
import click
import torch
import yaml
from addict import Dict
from libs.datasets import get_dataset
from libs.models import convnet
import numpy as np
from tensorboardX import SummaryWriter
from libs.models.ConditionalWGAN import ConditionalWGAN
from libs.models.AuxiliaryClassifierWGAN import AuxiliaryClassifierWGAN
from libs.models.ConditionalFCWGAN import ConditionalFCWGAN
from libs.models.AugNet import AugNet
from libs.utils.plots import plot_learning_curves
# from libs.models.InfoGAN import InfoGAN
# from libs.models.InfoWGAN import InfoWGAN

from libs.datasets.autism import data_prepare_aut
from libs.datasets.age import data_prepare_age
from hyperparam_logger import save_params
import os
from ast import literal_eval as make_tuple


@click.command()
@click.option("-c", "--config", type=str, required=True)
@click.option("--run_name", type=str, default='debug')
@click.option("--mode", type=str, default="-g-")
def main(config, run_name, mode):
    cuda = torch.cuda.is_available()
    cuda = cuda and torch.cuda.is_available()

    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on", torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")

    # Configuration
    config = Dict(yaml.load(open(config)))
    config.run_name = run_name

    # # Datasets
    # if config.DATASET == 'autism':
    #     data_prepare = data_prepare_aut
    # elif config.DATASET == 'age_small':
    #     data_prepare = data_prepare_age
    # else:
    #     raise NotImplementedError

    if mode == '---':
        cnn = convnet.ConnectomeConvNet(
            config,
            mix=False,
            val_loader=None,
            rnd_init=True
        ).cuda()
        path = config.CLASS_FROZEN_INIT_FILENAME
        torch.save(cnn.state_dict(), path)
        print("Random init frozen to file " + path + ".")


    # data_frame = data_prepare.prepare_dataset(config)
    train_dataset = get_dataset(config.DATASET)(
        config=config,
        split='train')

    val_dataset = get_dataset(config.DATASET)(
        config=config,
        split='val')

    test_dataset = get_dataset(config.DATASET)(
        config=config,
        split='test')

    # Data loaders
    train_class_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE.CLASS.TRAIN,
        num_workers=config.NUM_WORKERS.CLASS,
        shuffle=True,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE.CLASS.VAL,
        num_workers=config.NUM_WORKERS.CLASS,
        shuffle=False,
        pin_memory=True
    )




    if mode == 'c--':
        cnn = convnet.ConnectomeConvNet(
            config,
            mix=False,
            val_loader=val_loader,
        ).cuda()

        best_val = cnn.run_train(train_class_loader)
        print('Best valid loss: ' + str(best_val))
        save_params(config, {'validation_loss': best_val, 'mix_type': 'pure'})


    if mode == '-g-':
        gan = eval(config.GAN_MODEL)(config, train_dataset, val_loader)
        gan.netg.cuda()
        gan.netd.cuda()
        best_val = gan.train()
        print('Best valid loss: ' + str(best_val))
        save_params(config, {'validation_loss': best_val, 'mix_type': 'mixed'})

    if mode == 'exp':
        npz_filename = 'result_backup.npz'
        log_dir = os.path.join('runs', config.DATASET, config.RUN_NAME)
        writer = SummaryWriter(log_dir)
        pure_cnn_val_losses = []
        pure_cnn_dev_losses = []
        mixed_cnn_val_losses = []
        mixed_cnn_dev_losses = []
        train_sizes = make_tuple(config.EXP_TRAIN_SIZES)
        val_sizes = make_tuple(config.EXP_VAL_SIZES)
        dev_dataset = get_dataset(config.DATASET)(
            config=config,
            number_of_examples=config.EXP_DEV_SIZE,
            split='test'
        )
        dev_loader = torch.utils.data.DataLoader(
            dataset=dev_dataset,
            batch_size=config.BATCH_SIZE.CLASS.VAL,
            num_workers=config.NUM_WORKERS.CLASS,
            shuffle=False,
            pin_memory=True
        )
        for size_idx in range(len(train_sizes)):
            train_dataset = get_dataset(config.DATASET)(
                config=config,
                number_of_examples=train_sizes[size_idx],
                split='train')
            val_dataset = get_dataset(config.DATASET)(
                config=config,
                number_of_examples = val_sizes[size_idx],
                split='val')

            train_class_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=config.BATCH_SIZE.CLASS.TRAIN,
                num_workers=config.NUM_WORKERS.CLASS,
                shuffle=True,
                pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset,
                batch_size=config.BATCH_SIZE.CLASS.VAL,
                num_workers=config.NUM_WORKERS.CLASS,
                shuffle=False,
                pin_memory=True
            )

            num_cnn_runs = max(config.ITER_MAX.GAN // config.ITER_VAL.GAN,5)
            pure_cnn_val_sublist = []
            pure_cnn_dev_sublist = []
            for n in range(num_cnn_runs):
                cnn = convnet.ConnectomeConvNet(
                    config,
                    exp_name="pure_"+str(train_sizes[size_idx]+val_sizes[size_idx])+"_"+str(n),
                    val_loader=val_loader,
                ).cuda()

                best_val = cnn.run_train(train_class_loader)
                dev_error = cnn.evaluate(dev_loader)
                pure_cnn_val_sublist.append(best_val)
                pure_cnn_dev_sublist.append(dev_error)

            pure_cnn_val_losses.append(pure_cnn_val_sublist)
            pure_cnn_dev_losses.append(pure_cnn_dev_sublist)

            gan = eval(config.GAN_MODEL)(config, train_dataset, val_loader, )
            gan.netg.cuda()
            gan.netd.cuda()
            best_val = gan.train()

            gan_dev_err = gan.cnn.evaluate(dev_loader)
            mixed_cnn_dev_losses.append(gan_dev_err)
            mixed_cnn_val_losses.append(best_val)
            np.savez(
                os.path.join(gan.log_dir,npz_filename),
                pure_val=pure_cnn_val_losses,
                pure_dev=pure_cnn_dev_losses,
                mix_val=mixed_cnn_val_losses,
                mix_dev=mixed_cnn_dev_losses,
                last_size=str(train_sizes[size_idx]+val_sizes[size_idx])
            )
        print("Best pure val losses: ", pure_cnn_val_losses)
        print("Best mixed val losses: ", mixed_cnn_val_losses)
        print("Best pure dev losses: ", pure_cnn_dev_losses)
        print("Best mixed dev losses: ", mixed_cnn_dev_losses)
        save_params(config, {
            'pure_val_losses': str(pure_cnn_val_losses),
            'mix_val_losses': str(mixed_cnn_val_losses),
            'pure_dev_losses': str(pure_cnn_dev_losses),
            'mix_dev_losses': str(mixed_cnn_dev_losses),
        })
        plot_learning_curves(
            train_sizes,
            val_sizes,
            pure_cnn_val_losses,
            mixed_cnn_val_losses,
            pure_cnn_dev_losses,
            mixed_cnn_dev_losses,
            os.path.join(os.path.join('runs', config.DATASET, config.RUN_NAME, 'plots'))
        )

    if mode == '--c':
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=config.BATCH_SIZE.CLASS.TEST,
            num_workers=config.NUM_WORKERS.CLASS,
            shuffle=False,
        )

        cnn_test = convnet.ConnectomeConvNet(
            config,
            mix=False,
            val_loader=val_loader,
        ).cuda()
        ckpt_path = os.path.join(cnn_test.log_dir, 'checkpoint_best_class_mix_loss.pth')
        testloss = cnn_test.evaluate(path=ckpt_path, test_loader=test_loader)
        save_params(config, {'test_loss': testloss, 'mix_type': 'pixed'})


if __name__ == "__main__":
    main()
