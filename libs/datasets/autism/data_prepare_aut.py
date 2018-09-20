"""
@ author István Hajdu at MTA TTK
https://github.com/hajduistvan/connectome_gan
"""

import numpy as np
# import matplotlib.pyplot as plt

# import torch
import os


# def to_onehot(labels, num_classes=2):
#     one_hot = np.zeros((labels.shape[0], num_classes))
#     one_hot[np.arange(labels.shape[0]), labels] = 1
#     return one_hot.astype(np.float32)


def partition_data(relative_data_folder, data_source, limits, filename):
    cwd = os.getcwd()
    with np.load(os.path.join(cwd, relative_data_folder, data_source)) as df:
        data = np.transpose(df['data'], [0, 3, 1, 2]).astype(np.float32)
        labels = df['labels'].astype(np.float32)
    database_size = data.shape[0]
    idx = np.random.permutation(database_size)
    idx_train = idx[:int(database_size * limits[0])]
    idx_valid = idx[int(database_size * limits[0]):int(database_size * limits[1])]
    idx_test = idx[int(database_size * limits[1]):]

    train_data = data[idx_train]
    train_labels = labels[idx_train]
    valid_data = data[idx_valid]
    valid_labels = labels[idx_valid]
    test_data = data[idx_test]
    test_labels = labels[idx_test]
    # print(train_labels.shape)
    np.savez(filename, train_data=train_data, train_labels=train_labels,
             valid_data=valid_data, valid_labels=valid_labels,
             test_data=test_data, test_labels=test_labels)


def read_datasets(relative_data_folder, filename):
    cwd = os.getcwd()
    with np.load(os.path.join(cwd, relative_data_folder, filename)) as df:
        train_data = df['train_data']
        train_labels = df['train_labels']
        valid_data = df['valid_data']
        valid_labels = df['valid_labels']
        test_data = df['test_data']
        test_labels = df['test_labels']
        return {'train': (train_data, train_labels),
                'val': (valid_data, valid_labels),
                'test': (test_data, test_labels)}


def get_data_stats(dataset):
    class_0_mean = []
    class_1_mean = []
    class_0_std = []
    class_1_std = []
    class_0_idx = []
    class_1_idx = []

    data = dataset[0]
    labels = dataset[1]
    for i in range(len(data)):
        if labels[i] == 0:
            class_0_mean.append(np.mean(data[i]))
            class_0_std.append(np.std(data[i]))
            class_0_idx.append(i)

        else:
            class_1_mean.append(np.mean(data[i]))
            class_1_std.append(np.std(data[i]))
            class_1_idx.append(i)
    return {'mean0': class_0_mean,
            'mean1': class_1_mean,
            'std0': class_0_std,
            'std1': class_1_std,
            'idx0': class_0_idx,
            'idx1': class_1_idx}


# def plot_histograms(data_stats, part, bins):
#     plt.subplot(221)
#     plt.hist(data_stats['mean0'], bins)
#     plt.title(part + ' dataset class 0 means')
#     plt.subplot(222)
#     plt.hist(data_stats['mean1'], bins)
#     plt.title(part + ' dataset class 1 means')
#     plt.subplot(223)
#     plt.hist(data_stats['std0'], bins)
#     plt.title(part + ' dataset class 0 stds')
#     plt.subplot(224)
#     plt.hist(data_stats['std1'], bins)
#     plt.title(part + ' dataset class 1 stds')
#     plt.show()


def calc_outlie_dist(data_stats, alpha):
    c10m_m = np.mean(data_stats['mean0'])
    cl1m_m = np.mean(data_stats['mean1'])
    cl0s_m = np.mean(data_stats['std0'])
    cl1s_m = np.mean(data_stats['std1'])
    id0 = data_stats['idx0']
    id1 = data_stats['idx1']
    out_0 = []
    out_1 = []
    for i in range(len(id0)):
        out_0.append((1 - alpha) * (data_stats['mean0'][i] - c10m_m) ** 2 +
                     alpha * (data_stats['std0'][i] - cl0s_m) ** 2)

    for i in range(len(id1)):
        out_1.append((1 - alpha) * (data_stats['mean1'][i] - cl1m_m) ** 2 +
                     alpha * (data_stats['std1'][i] - cl1s_m) ** 2)
    return out_0, out_1


def mark_outliers(data_stats, out_0, out_1, threshhold):
    cl0_o_id = []
    cl1_o_id = []
    ok_0_id = []
    ok_1_id = []
    for i in range(len(out_0)):
        if out_0[i] > threshhold:
            cl0_o_id.append(data_stats['idx0'][i])
        else:
            ok_0_id.append(data_stats['idx0'][i])

    for i in range(len(out_1)):
        if out_1[i] > threshhold:
            cl1_o_id.append(data_stats['idx1'][i])
        else:
            ok_1_id.append(data_stats['idx1'][i])
    return cl0_o_id, cl1_o_id, ok_0_id, ok_1_id


def make_ok_dataset(dataset, ok_id_0, ok_id_1):
    data = np.zeros((len(ok_id_0) + len(ok_id_1),
                     dataset[0].shape[1],
                     dataset[0].shape[2],
                     dataset[0].shape[3]))
    labels = np.zeros((len(ok_id_0) + len(ok_id_1)))
    for i in range(len(ok_id_0)):
        data[i] = dataset[0][ok_id_0[i]]
        labels[i] = dataset[1][ok_id_0[i]]
    for i in range(len(ok_id_0), len(ok_id_0) + len(ok_id_1)):
        data[i] = dataset[0][ok_id_1[i - len(ok_id_0)]]
        labels[i] = dataset[1][ok_id_1[i - len(ok_id_0)]]
    return (data.astype(np.float32), labels.astype(np.float32))


def prepare_dataset(CONFIG):
    if not os.path.isfile(os.path.join(
            CONFIG.RELATIVE_DATA_FOLDER,
            CONFIG.DATASET_PART)):
        partition_data(
            CONFIG.RELATIVE_DATA_FOLDER,
            CONFIG.DATASET_WHOLE,
            CONFIG.LIMITS,
            CONFIG.DATASET_PART)
        print("Data partitioned")
    else:
        print("Data already partitioned")
    data_frame = read_datasets(
        CONFIG.RELATIVE_DATA_FOLDER,
        CONFIG.DATASET_PART)
    train_data_stats = get_data_stats(data_frame['train'])
    out_0, out_1 = calc_outlie_dist(
        train_data_stats, CONFIG.OUTLY_MIXER)
    out_id_0, out_id_1, ok_id_0, ok_id_1 = mark_outliers(
        train_data_stats, out_0, out_1, CONFIG.OUTLY_THRESHOLD)
    train_dataset_ok = make_ok_dataset(
        data_frame['train'],
        ok_id_0, ok_id_1)
    return {'train': train_dataset_ok,
            'val': data_frame['val'],
            'test': data_frame['test']}
