# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:26:36 2018

@author: hajduistvan
"""

import numpy as np
import torch.nn as nn
import torch
from tensorboardX import SummaryWriter
import os
from ast import literal_eval as make_tuple


# from torchnet.meter import MovingAverageValueMeter

def get_loss_denorm_fn(CONFIG):
    if CONFIG.SUPERVISE_TYPE == 'regress':
        age_m = make_tuple(CONFIG.AGE_INTERVAL)[0]
        age_M = make_tuple(CONFIG.AGE_INTERVAL)[1]
        return lambda x: x * (age_M - age_m)
    else:
        return lambda x: x

def evaluate_classifier(net, test_loader, CONFIG, mix, early):
    loss_denorm_fn = get_loss_denorm_fn(CONFIG)

    if mix:
        log_dir = CONFIG.LOG_DIR.CLASS.MIXED
        save_dir = CONFIG.SAVE_DIR.CLASS.MIXED
        _str1 = 'mixed'
    else:
        log_dir = CONFIG.LOG_DIR.CLASS.PURE
        save_dir = CONFIG.SAVE_DIR.CLASS.PURE
        _str1 = 'pure'
    writer = SummaryWriter(log_dir)
    if early == 'loss':
        net.load_state_dict(torch.load(os.path.join(save_dir, "checkpoint_best_class_loss.pth")))
        _str2 = 'With loss-based'
        _str3 = '_early'
    elif early == 'acc':
        net.load_state_dict(torch.load(os.path.join(save_dir, "checkpoint_best_class_acc.pth")))
        _str2 = 'With accuracy-based'
        _str3 = '_early'
    else:
        _str2 = 'Without'
        _str3 = '_last_iter'

    net.eval()
    if CONFIG.SUPERVISE_TYPE == 'binary':
        criterion = nn.CrossEntropyLoss().cuda()
    elif CONFIG.SUPERVISE_TYPE == 'regress':
        criterion = nn.L1Loss().cuda()
    else:
        raise NotImplementedError()
    l, a, b = 0, 0, 0
    for i, datapair in enumerate(test_loader):
        with torch.no_grad():
            data, labels = datapair
            batch_len = data.shape[0]
            data = data.cuda()
            if CONFIG.SUPERVISE_TYPE == 'binary':
                labels = labels.type(torch.long).view(-1).cuda()
            else:
                labels = labels.type(torch.FloatTensor).view(-1, 1).cuda()
            output = net(data)
            loss = criterion(output, labels)
            ll = loss.detach().cpu().numpy()
            l += ll * batch_len
            b += batch_len
            if CONFIG.SUPERVISE_TYPE == 'binary':
                max_index = output.max(dim=1)[1]
                accuracy = (max_index == labels).sum() * 100 / labels.shape[0]
                acc = accuracy.detach().cpu().numpy()
                a += acc * batch_len

    testloss  = loss_denorm_fn(l / b)
    writer.add_scalar('TEST_LOSS' + _str3, testloss)

    if CONFIG.SUPERVISE_TYPE == 'binary':
        testacc = a / b
        writer.add_scalar('TEST_LOSS' + _str3, testacc)



    print('===== ' + _str2 + ' early stopping =====\n\nTest loss ' + _str1 + ' :',
          np.around(testloss, 5))
    dict = {'test_loss': testloss}
    if CONFIG.SUPERVISE_TYPE == 'binary':
        print('Test acc ' + _str1 + ' :', testacc, '\n')
        dict = {**dict, 'test_acc': testacc}

    return dict
