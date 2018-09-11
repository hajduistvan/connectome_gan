# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:26:36 2018

@author: hajduistvan
"""

# External imports
import numpy as np
import torch
import torchvision.utils as vutils
import torch.nn as nn
# from torch.utils.data import DataLoader
from ast import literal_eval as make_tuple

# Internal imports
# import local_networks as nets
from lib.models import wgangp_fns
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torchnet.meter import MovingAverageValueMeter

import os


def get_loss_denorm_fn(CONFIG):
    if CONFIG.SUPERVISE_TYPE == 'regress':
        age_m = make_tuple(CONFIG.AGE_INTERVAL)[0]
        age_M = make_tuple(CONFIG.AGE_INTERVAL)[1]
        return lambda x: x * (age_M - age_m)
    else:
        return lambda x: x


# Connectome classifier

def train_classifier(net, CONFIG, trainloader, validloader, device, mix):
    # Criterion (classifier!!!)
    loss_denorm_fn = get_loss_denorm_fn(CONFIG)
    net.train()
    if mix:
        save_dir = CONFIG.SAVE_DIR.CLASS.MIXED
        log_dir = CONFIG.LOG_DIR.CLASS.MIXED
        _str1 = 'mixed'
    else:
        save_dir = CONFIG.SAVE_DIR.CLASS.PURE
        log_dir = CONFIG.LOG_DIR.CLASS.PURE
        _str1 = 'pure'
    os.makedirs(save_dir, exist_ok=True)
    if CONFIG.SUPERVISE_TYPE == 'binary':
        criterion = nn.CrossEntropyLoss().cuda()
    elif CONFIG.SUPERVISE_TYPE == 'regress':
        criterion_to_log = nn.L1Loss().cuda()
        if CONFIG.ARCHITECTURE.CLASS.LOSS_FN == 'MSE':
            criterion = nn.MSELoss().cuda()
        elif CONFIG.ARCHITECTURE.CLASS.LOSS_FN == 'L1':
            criterion = nn.L1Loss().cuda()
        elif CONFIG.ARCHITECTURE.CLASS.LOSS_FN == 'SmoothL1':
            criterion = nn.SmoothL1Loss().cuda()
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    best_valid_loss = np.inf
    if CONFIG.SUPERVISE_TYPE == 'binary':
        best_valid_acc_from_loss = 0
        best_valid_acc = 0
        best_valid_loss_from_acc = np.inf
    optimizer = {
        "adam": torch.optim.Adam(
            net.parameters(),
            lr=float(CONFIG.OPTIMIZER.CLASS.LR_ADAM),
            betas=make_tuple(CONFIG.OPTIMIZER.CLASS.BETAS),
        )
    }[CONFIG.OPTIMIZER.CLASS.ALG]

    trainiter = iter(trainloader)

    writer = SummaryWriter(log_dir)
    if CONFIG.SUPERVISE_TYPE == 'binary':
        train_acc_meter = MovingAverageValueMeter(5)
        val_acc_from_loss_meter = MovingAverageValueMeter(5)

    train_loss_meter = MovingAverageValueMeter(5)
    val_loss_meter = MovingAverageValueMeter(5)

    for iteration in tqdm(
            range(1, CONFIG.ITER_MAX.CLASS + 1),
            total=CONFIG.ITER_MAX.CLASS,
            leave=False,
            dynamic_ncols=True,
    ):
        try:
            data, labels = trainiter.next()
        except StopIteration:
            trainiter = iter(trainloader)
            data, labels = trainiter.next()
        if CONFIG.SUPERVISE_TYPE == 'binary':
            labels = labels.type(torch.long).view(-1)
        else:
            labels = labels.type(torch.FloatTensor).view(-1, 1)

        data, labels = data.cuda(), labels.cuda()

        optimizer.zero_grad()

        outputs = net(data)

        loss = criterion(outputs, labels)
        if CONFIG.SUPERVISE_TYPE == 'regress':
            loss_log = criterion_to_log(outputs, labels).detach().cpu()
        loss.backward()
        optimizer.step()

        if CONFIG.SUPERVISE_TYPE == 'binary':
            max_index = outputs.max(dim=1)[1]
            accuracy = (max_index == labels).sum() * 100 / labels.shape[0]
            accuracy = accuracy.type(torch.float)
            train_acc_meter.add(accuracy.detach().cpu())
            train_loss_meter.add(loss_denorm_fn(loss.detach().cpu()))
        else:
            train_loss_meter.add(loss_denorm_fn(loss_log))

        # train_loss_meter.add(loss_denorm_fn(loss.detach().cpu()))

        if iteration % CONFIG.ITER_TB.CLASS == 0:
            writer.add_scalar("class_train_loss", train_loss_meter.value()[0], iteration)
            if CONFIG.SUPERVISE_TYPE == 'binary':
                writer.add_scalar("class_train_acc", train_acc_meter.value()[0], iteration)
        if iteration % CONFIG.ITER_SAVE.CLASS == 0:
            torch.save(
                net.state_dict(),
                os.path.join(save_dir,
                             "checkpoint_class_{}.pth".format(iteration)),
            )

        # Validation
        if iteration % CONFIG.ITER_VAL == 0:
            net.eval()
            if CONFIG.SUPERVISE_TYPE == 'binary':
                valid_acc_l, = torch.cuda.FloatTensor(1).fill_(0)
            valid_loss_l = torch.cuda.FloatTensor(1).fill_(0)
            b = 0
            with torch.no_grad():
                for i, datapair in enumerate(validloader):
                    data, labels = datapair
                    batch_len = data.shape[0]
                    if CONFIG.SUPERVISE_TYPE == 'binary':
                        labels = labels.type(torch.long).view(-1)
                    else:
                        labels = labels.type(torch.FloatTensor).view(-1, 1)
                    data, labels = data.cuda(), labels.cuda()
                    out = net(data)

                    loss = loss_denorm_fn(criterion_to_log(out, labels).type(torch.float).cuda().detach())
                    # print(loss)
                    # l = loss.detach().cpu().numpy()
                    if CONFIG.SUPERVISE_TYPE == 'binary':
                        max_index = out.max(dim=1)[1]
                        accuracy = (max_index == labels).sum() * 100 / labels.shape[0]
                        accuracy = accuracy.type(torch.float).detach()
                        valid_acc_l += accuracy * batch_len
                    # acc = accuracy.detach().cpu().numpy()
                    # print('sima', loss, accuracy)

                    valid_loss_l += loss * batch_len
                    b += batch_len
            if CONFIG.SUPERVISE_TYPE == 'binary':
                valid_acc = (valid_acc_l / b).type(torch.float).cpu()
                val_acc_from_loss_meter.add(valid_acc)
                writer.add_scalar("val_acc", val_acc_from_loss_meter.value()[0], iteration)

            valid_loss = (valid_loss_l / b).type(torch.float).cpu()
            val_loss_meter.add(valid_loss)

            # print(valid_loss, valid_acc)
            writer.add_scalar("val_loss", val_loss_meter.value()[0], iteration)

            if valid_loss < best_valid_loss:
                torch.save(
                    net.state_dict(),
                    os.path.join(save_dir,
                                 "checkpoint_best_class_loss.pth"),
                )
                best_valid_loss = valid_loss
                if CONFIG.SUPERVISE_TYPE == 'binary':
                    best_valid_acc_from_loss = valid_acc

            if CONFIG.SUPERVISE_TYPE == 'binary' and valid_acc > best_valid_acc:
                torch.save(
                    net.state_dict(),
                    os.path.join(save_dir,
                                 "checkpoint_best_class_acc.pth"),
                )
                best_valid_loss_from_acc = valid_loss
                best_valid_acc = valid_acc

            net.train()

    dict = {
        'VAL_LOSS': best_valid_loss.detach().cpu().data.numpy()[0],
    }
    # print('CNN training finished')
    print('Best valid loss ' + _str1 + ': ', best_valid_loss.detach().cpu().data.numpy()[0])
    if CONFIG.SUPERVISE_TYPE == 'binary':
        print('Best valid acc from loss' + _str1 + ': ', best_valid_acc_from_loss.detach().cpu().data.numpy(), )
        print('Best valid loss from acc ' + _str1 + ': ', best_valid_loss_from_acc[0].cpu().data.numpy(), )
        print('Best valid acc ' + _str1 + ': ', best_valid_acc.detach().cpu().data.numpy(), )
        dict = {**dict, 'VAL_ACC': best_valid_acc_from_loss.detach().cpu().data.numpy()}

    return dict


# Connectome Conditional WGAN-BP


def train_cond_wgan_gp(netD, netG, CONFIG, train_loader):
    # rnd labels generator
    rnd_lab_gen_fn = wgangp_fns.get_rnd_label_generator(CONFIG)

    # Optimizer
    optimizerD = {
        "adam": torch.optim.Adam(
            netD.parameters(),
            lr=float(CONFIG.OPTIMIZER.GAN.D.LR_ADAM),
            betas=make_tuple(CONFIG.OPTIMIZER.GAN.D.BETAS),
        ),
        "sgd": torch.optim.SGD( # Do not use!
            netD.parameters(),
            lr=float(CONFIG.OPTIMIZER.GAN.D.LR_SGD),
            momentum=CONFIG.OPTIMIZER.GAN.D.MOMENTUM,
            nesterov=False,
        ),
        "rmsprop": torch.optim.RMSprop(
            netD.parameters(),
            lr=float(CONFIG.OPTIMIZER.GAN.D.LR_RMS),
            momentum=CONFIG.OPTIMIZER.GAN.D.MOMENTUM,
        ),
    }[CONFIG.OPTIMIZER.GAN.D.ALG]
    optimizerG = {
        "adam": torch.optim.Adam(
            netG.parameters(),
            lr=float(CONFIG.OPTIMIZER.GAN.G.LR_ADAM),
            betas=make_tuple(CONFIG.OPTIMIZER.GAN.G.BETAS),
        ),
        "sgd": torch.optim.SGD( # Do not use!
            netG.parameters(),
            lr=float(CONFIG.OPTIMIZER.GAN.G.LR_SGD),
            momentum=CONFIG.OPTIMIZER.GAN.G.MOMENTUM,
            nesterov=False,
        ),
        "rmsprop": torch.optim.RMSprop(
            netG.parameters(),
            lr=float(CONFIG.OPTIMIZER.GAN.G.LR_RMS),
            momentum=CONFIG.OPTIMIZER.GAN.G.MOMENTUM,
        ),
    }[CONFIG.OPTIMIZER.GAN.G.ALG]

    dataiter = iter(train_loader)

    writer = SummaryWriter(CONFIG.LOG_DIR.GAN)
    w_loss_meter = MovingAverageValueMeter(5)
    d_loss_meter = MovingAverageValueMeter(5)
    r_loss_meter = MovingAverageValueMeter(5)
    f_loss_meter = MovingAverageValueMeter(5)
    gp_loss_meter = MovingAverageValueMeter(5)
    g_loss_meter = MovingAverageValueMeter(5)

    for iteration in tqdm(
            range(1, CONFIG.ITER_MAX.GAN + 1),
            total=CONFIG.ITER_MAX.GAN,
            leave=False,
            dynamic_ncols=True,
    ):
        # =====================================================================
        # (1) Update Discriminator network
        # =====================================================================
        for p in netD.parameters():
            p.requires_grad = True

        for iter_d in range(CONFIG.ARCHITECTURE.GAN.CRITIC_ITERS):
            try:
                _data, _labels = dataiter.next()
            except StopIteration:
                # print('iteration stopped, restarted')
                dataiter = iter(train_loader)
                _data, _labels = dataiter.next()

            real_data = torch.Tensor(_data).cuda()
            real_labels = torch.Tensor(_labels).cuda()

            netD.zero_grad()

            # train with real
            d_real = netD(real_data, real_labels)[0].mean()

            # train with fake
            noise = torch.randn(
                real_data.shape[0],
                CONFIG.ARCHITECTURE.GAN.NOISE_DIMS
            ).cuda()  # .requires_grad_(False)

            fake = netG(noise, real_labels).data

            d_fake = netD(fake, real_labels)[0].mean()

            fake_sq = fake.view(-1, 1, CONFIG.MATR_SIZE, CONFIG.MATR_SIZE)

            # train with gradient penalty
            gradient_penalty = wgangp_fns.calc_gradient_penalty_cond(
                netD,
                real_data.data,
                real_labels,
                fake_sq.data
            )

            # consistency_cost
            # ct_cost = wgangp_fns.calc_consistency_penalty(
            #     netD,
            #     real_data.data,
            #     real_labels,
            #     CONFIG.ARCHITECTURE.GAN.CT_M
            # )
            d_cost = d_fake - d_real + gradient_penalty * CONFIG.ARCHITECTURE.GAN.LAMBDA_GP# \
                                     #+ ct_cost * CONFIG.ARCHITECTURE.GAN.LAMBDA_CT
            wasserstein_d = d_real - d_fake

            d_cost.backward()

            w_loss_meter.add(wasserstein_d.detach().cpu())
            d_loss_meter.add(d_cost.detach().cpu())
            r_loss_meter.add(d_real.detach().cpu())
            f_loss_meter.add(d_fake.detach().cpu())
            gp_loss_meter.add(gradient_penalty.detach().cpu())

            optimizerD.step()

        ############################
        # (2) Update G network
        ###########################
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()

        fake_labels = rnd_lab_gen_fn().cuda()

        noise = torch.randn(
            CONFIG.BATCH_SIZE.GAN.TRAIN,
            CONFIG.ARCHITECTURE.GAN.NOISE_DIMS
        ).cuda()
        fake = netG(noise, fake_labels)
        G  = netD(fake, fake_labels)
        G = G.mean()

        g_cost = -G
        g_cost.backward()
        optimizerG.step()
        g_loss_meter.add(g_cost.detach().cpu())

        # TensorBoard
        if iteration % CONFIG.ITER_TB.GAN == 0:
            writer.add_scalar("wasserstein_loss", w_loss_meter.value()[0], iteration)
            writer.add_scalar("discriminator_loss", d_loss_meter.value()[0], iteration)
            writer.add_scalar("disc_real_loss", r_loss_meter.value()[0], iteration)
            writer.add_scalar("disc_fake_loss", d_loss_meter.value()[0], iteration)
            writer.add_scalar("gradient_penalty", gp_loss_meter.value()[0], iteration)
            writer.add_scalar("generator_loss", g_loss_meter.value()[0], iteration)

        if iteration % CONFIG.ITER_SAVE.GAN == 0:
            os.makedirs(CONFIG.SAVE_DIR.GAN, exist_ok=True)
            torch.save(
                netD.state_dict(),
                os.path.join(CONFIG.SAVE_DIR.GAN, "checkpoint_disc_{}.pth".format(iteration)),
            )
            torch.save(
                netG.state_dict(),
                os.path.join(CONFIG.SAVE_DIR.GAN, "checkpoint_gen_{}.pth".format(iteration)),
            )
            gen_imgs, gen_labels = wgangp_fns.generate_image_cond(
                netG,
                CONFIG
            )
            wgangp_fns.save_generated_images(gen_imgs, gen_labels, CONFIG, iteration)
