import click
# import cv2
import torch
import yaml
from addict import Dict
from libs.datasets import get_dataset
from libs.models import convnet, WGAN, ACWGAN
from libs.datasets.autism import data_prepare_aut
from libs.datasets.age import data_prepare_age
from hyperparam_logger import save_params

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

    # Datasets
    if config.DATASET == 'autism':
        data_prepare = data_prepare_aut
    elif config.DATASET == 'age_small':
        data_prepare = data_prepare_age
    else:
        raise NotImplementedError

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


    data_frame = data_prepare.prepare_dataset(config)
    train_dataset = get_dataset(config.DATASET)(
        data_frame,
        config=config,
        split='train')

    val_dataset = get_dataset(config.DATASET)(
        data_frame,
        config=config,
        split='val')

    test_dataset = get_dataset(config.DATASET)(
        data_frame,
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

        gan = ACWGAN.ACWGAN(config, train_dataset, val_loader)
        gan.netg.cuda()
        gan.netd.cuda()
        best_val = gan.train()
        print('Best valid loss: ' + str(best_val))
        save_params(config, {'validation_loss': best_val, 'mix_type': 'mixed'})


    if mode == '--c':
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=config.BATCH_SIZE.CLASS.TEST,
            num_workers=config.NUM_WORKERS.CLASS,
            shuffle=False,
        )
        cnn_test = convnet.ConnectomeConvNet(
            config,
            mix=True,
            val_loader=val_loader,
        ).cuda()
        mixed_dataset = None #TODO: !!!
        cnn_test.run_train(mixed_dataset)

    # # -----------------------------------------
    # # CNN PURE
    # # -----------------------------------------
    # if mode[0] == 'c':
    #     print('Starting training pure cnn\n')
    #     res_dict_val = train_loops.train_classifier(
    #         cnn,
    #         config,
    #         train_real_class_loader,
    #         val_loader,
    #         device,
    #         False
    #     )
    #     evaluate.evaluate_classifier(
    #         cnn, test_loader, config, mix=False, early=None)
    #     res_dict_test = evaluate.evaluate_classifier(
    #         cnn, test_loader, config, mix=False, early='loss')
    #     if config.SUPERVISE_TYPE == 'binary':
    #         evaluate.evaluate_classifier(
    #             cnn, test_loader, config, mix=False, early='acc')
    #     result_dict = {**res_dict_test, **res_dict_val, 'MIX_TYPE': 'pure'}
    #     save_params(config, result_dict)
    #
    #
    # # -----------------------------------------
    # # GAN
    # # ------------------------------------------
    # if mode[1] == 'g':
    #     print('Starting training gan\n')
    #     train_loops.train_cond_wgan_gp(
    #         netD,
    #         netG,
    #         config,
    #         train_real_gan_loader
    #     )
    #
    # # Exctracting generated images
    # if mode[2] == 'c':
    #     print('Extracting gen images\n')
    #     if mode[1] == '-':
    #         netG = ACWGAN.Generator(
    #             config.ARCHITECTURE.GAN.DIM,
    #             config.MATR_SIZE,
    #             config.ARCHITECTURE.GAN.NOISE_DIMS
    #         ).cuda()
    #         save_dir = config.SAVE_DIR.GAN
    #         load_it = str(config.GEN_EARLY_IT)
    #         netG.load_state_dict(torch.load(
    #             save_dir + '/checkpoint_gen_' + load_it + '.pth'))
    #         print('Loaded generator iter ' + load_it + '.')
    #     gen_dataset = wgangp_fns.generate_training_images(netG, config, device)
    #     train_mixed_dataset = get_dataset(config.DATASET)(
    #         data_frame, config=config, data_gen=gen_dataset, split='train')
    #     if not config.CLASS_FROZEN_INIT_FILENAME is None:
    #         cnn_mix.load_state_dict(torch.load(config.CLASS_FROZEN_INIT_FILENAME))
    #     train_mixed_class_loader = torch.utils.data.DataLoader(
    #         dataset=train_mixed_dataset,
    #         batch_size=config.BATCH_SIZE.CLASS.TRAIN,
    #         num_workers=config.NUM_WORKERS.CLASS,
    #         shuffle=True,
    #     )
    #
    #     # -----------------------------------------
    #     # CNN MIXED
    #     # ------------------------------------------
    #
    #     print('Starting training mixed cnn\n')
    #
    #     res_dict_val = train_loops.train_classifier(
    #         cnn_mix,
    #         config,
    #         train_mixed_class_loader,
    #         val_loader,
    #         device,
    #         True
    #     )
    #     evaluate.evaluate_classifier(
    #         cnn_mix, test_loader, config, mix=True, early=None)
    #     res_dict_test = evaluate.evaluate_classifier(
    #         cnn_mix, test_loader, config, mix=True, early='loss')
    #     if config.SUPERVISE_TYPE == 'binary':
    #         evaluate.evaluate_classifier(
    #             cnn_mix, test_loader, config, mix=True, early='acc')
    #     result_dict = {**res_dict_test, **res_dict_val, 'MIX_TYPE': 'mixed'}
    #     save_params(config, result_dict)


if __name__ == "__main__":
    main()
