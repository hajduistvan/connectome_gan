import click
# import cv2
import torch
import yaml
from addict import Dict
from libs.datasets import get_dataset
from libs.models import convnet, WGAN, wgangp_fns
from libs.datasets.autism import data_prepare_aut
from libs.datasets.age import data_prepare_age
import train_loops
import evaluate
from ast import literal_eval as make_tuple
import matplotlib.pyplot as plt
import numpy as np
import os
from hyperparam_logger import save_params

cuda = torch.cuda.is_available()
cuda = cuda and torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
config = 'config/conf_age_regr.yaml'
if cuda:
    current_device = torch.cuda.current_device()
    print("Running on", torch.cuda.get_device_name(current_device))
else:
    print("Running on CPU")

# Configuration
CONFIG = Dict(yaml.load(open(config)))

# Datasets
if CONFIG.DATASET == 'autism':
    data_prepare = data_prepare_aut
elif CONFIG.DATASET == 'age':
    data_prepare = data_prepare_age
else:
    raise NotImplementedError

data_frame = data_prepare.prepare_dataset(CONFIG)
train_real_dataset = get_dataset(CONFIG.DATASET)(
    data_frame,
    CONFIG=CONFIG,
    data_gen=None,
    split='train')

val_dataset = get_dataset(CONFIG.DATASET)(
    data_frame,
    CONFIG=CONFIG,
    data_gen=None,
    split='val')

test_dataset = get_dataset(CONFIG.DATASET)(
    data_frame,
    CONFIG=CONFIG,
    data_gen=None,
    split='test')

# Data loaders
train_real_class_loader = torch.utils.data.DataLoader(
    dataset=train_real_dataset,
    batch_size=CONFIG.BATCH_SIZE.CLASS.TRAIN,
    num_workers=CONFIG.NUM_WORKERS.CLASS,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=CONFIG.BATCH_SIZE.CLASS.VAL,
    num_workers=CONFIG.NUM_WORKERS.CLASS,
    shuffle=False,
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=CONFIG.BATCH_SIZE.CLASS.TEST,
    num_workers=CONFIG.NUM_WORKERS.CLASS,
    shuffle=False,
)
train_real_gan_loader = torch.utils.data.DataLoader(
    dataset=train_real_dataset,
    batch_size=CONFIG.BATCH_SIZE.GAN.TRAIN,
    num_workers=CONFIG.NUM_WORKERS.GAN,
    shuffle=True,
)
#%%
mode = 'c-c'
if mode[0] == 'c':
    cnn = convnet.ConnectomeConvNet(
        CONFIG.MATR_SIZE,
        make_tuple(CONFIG.ARCHITECTURE.CLASS.CHS),
        make_tuple(CONFIG.ARCHITECTURE.CLASS.NEURONS),
        CONFIG.ARCHITECTURE.CLASS.DROPOUT,
        make_tuple(CONFIG.ARCHITECTURE.CLASS.DROPOUT_LAYERS),
        make_tuple(CONFIG.ARCHITECTURE.CLASS.BN_LAYERS),
    ).cuda()
if mode[1] == 'g':
    netG = WGAN.Generator(
        CONFIG.ARCHITECTURE.GAN.DIM,
        CONFIG.MATR_SIZE,
        CONFIG.ARCHITECTURE.GAN.NOISE_DIMS,
        # CONFIG.ARCHITECTURE.GAN.DROPOUT
    ).cuda()
    netD = WGAN.Discriminator(
        CONFIG.ARCHITECTURE.GAN.DIM,
        CONFIG.MATR_SIZE,
        # CONFIG.ARCHITECTURE.GAN.DROPOUT
    ).cuda()
    # print(netG)
    # print(netD)
    # print(netG.fc1_labels.)#, netG.fc1_labels)
if mode[2] == 'c':
    cnn_mix = convnet.ConnectomeConvNet(
        CONFIG.MATR_SIZE,
        make_tuple(CONFIG.ARCHITECTURE.CLASS.CHS),
        make_tuple(CONFIG.ARCHITECTURE.CLASS.NEURONS),
        CONFIG.ARCHITECTURE.CLASS.DROPOUT,
        make_tuple(CONFIG.ARCHITECTURE.CLASS.DROPOUT_LAYERS),
        make_tuple(CONFIG.ARCHITECTURE.CLASS.BN_LAYERS),
    ).cuda()
# -----------------------------------------
# CNN PURE
# -----------------------------------------
if mode[0] == 'c':
    print('Starting training pure cnn\n')
    res_dict_val = train_loops.train_classifier(
        cnn,
        CONFIG,
        train_real_class_loader,
        val_loader,
        device,
        False
    )
    evaluate.evaluate_classifier(
        cnn, test_loader, CONFIG, mix=False, early=None)
    res_dict_test = evaluate.evaluate_classifier(
        cnn, test_loader, CONFIG, mix=False, early='loss')
    if CONFIG.SUPERVISE_TYPE == 'binary':
        evaluate.evaluate_classifier(
            cnn, test_loader, CONFIG, mix=False, early='acc')
    result_dict = {**res_dict_test, **res_dict_val, 'MIX_TYPE': 'pure'}
    save_params(CONFIG, result_dict)

# -----------------------------------------
# GAN
# ------------------------------------------
if mode[1] == 'g':
    print('Starting training gan\n')
    train_loops.train_cond_wgan_gp(
        netD,
        netG,
        CONFIG,
        train_real_gan_loader
    )

# Exctracting generated images
if mode[2] == 'c':
    print('Extracting gen images\n')
    if mode[1] == '-':
        netG = WGAN.Generator(
            CONFIG.ARCHITECTURE.GAN.DIM,
            CONFIG.MATR_SIZE,
            CONFIG.ARCHITECTURE.GAN.NOISE_DIMS
        ).cuda()
        save_dir = CONFIG.SAVE_DIR.GAN
        load_it = str(CONFIG.GEN_EARLY_IT)
        netG.load_state_dict(torch.load(
            save_dir + '/checkpoint_gen_' + load_it + '.pth'))
        print('Loaded generator iter ' + load_it + '.')
    gen_dataset = wgangp_fns.generate_training_images(netG, CONFIG, device)
    train_mixed_dataset = get_dataset(CONFIG.DATASET)(
        data_frame, CONFIG=CONFIG, data_gen=gen_dataset, split='train')

    train_mixed_class_loader = torch.utils.data.DataLoader(
        dataset=train_mixed_dataset,
        batch_size=CONFIG.BATCH_SIZE.CLASS.TRAIN,
        num_workers=CONFIG.NUM_WORKERS.CLASS,
        shuffle=True,
    )

    # -----------------------------------------
    # CNN MIXED
    # ------------------------------------------

    print('Starting training mixed cnn\n')

    res_dict_val = train_loops.train_classifier(
        cnn_mix,
        CONFIG,
        train_mixed_class_loader,
        val_loader,
        device,
        True
    )
    evaluate.evaluate_classifier(
        cnn_mix, test_loader, CONFIG, mix=True, early=None)
    res_dict_test = evaluate.evaluate_classifier(
        cnn_mix, test_loader, CONFIG, mix=True, early='loss')
    if CONFIG.SUPERVISE_TYPE == 'binary':
        evaluate.evaluate_classifier(
            cnn_mix, test_loader, CONFIG, mix=True, early='acc')
    result_dict = {**res_dict_test, **res_dict_val, 'MIX_TYPE': 'mixed'}
    save_params(CONFIG, result_dict)

# %%
train_iter = iter(train_real_class_loader)
M, m, nanids = [], [], []
for i, data in enumerate(train_iter):
    img, lab = data
    # print(img_slice.shape)
    fig = plt.figure()
    plt.imshow(img[0,0,:,:], cmap='jet', interpolation='nearest', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('class ' + str(lab))
    plt.axis('off')
    fig.savefig(os.path.join(os.getcwd(), 'train_img_debug', str(i)+'.png'))
M = np.array(M)
m = np.array(m)
nansum = np.sum(m != m)
print('# of NaNs:', nansum)
print('NaN valued indeces: ',nanids)
# %%
