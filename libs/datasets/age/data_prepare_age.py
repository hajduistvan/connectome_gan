import os
import numpy as np
import pickle
import h5py
import csv


def partition_data(CONFIG):
    relative_data_folder = CONFIG.RELATIVE_DATA_FOLDER
    filename = relative_data_folder + CONFIG.DATASET_PART
    # test data
    test_data = []
    test_data_filename = relative_data_folder + 'inhouse/CORR_tensor_inhouse.pickle'
    with open(test_data_filename, 'rb') as f:
        test_df = pickle.load(f)
        for k, v in test_df.items():
            test_data.append(v)
    test_data = np.array(test_data)
    test_data = np.reshape(test_data, (-1, 1, CONFIG.MATR_SIZE, CONFIG.MATR_SIZE,))

    # test labels
    test_labels_filename = relative_data_folder + 'inhouse/labels_inhouse.txt'
    test_ids = []
    test_regr_labels = []
    test_class_labels = []
    with open(test_labels_filename, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for currentline in reader:
            test_ids.append(int(currentline[0]))
            test_regr_labels.append(int(currentline[2]))
            test_class_labels.append(int(currentline[1]))

    test_ids = np.array(test_ids)
    test_regr_labels = np.array(test_regr_labels)
    test_class_labels = np.array(test_class_labels)

    # valid data

    valid_data_filenames = [
        relative_data_folder + 'nki_rs_en/subdat_1_nkiRSen_resting_harvardoxfordfull_111ROIs.mat',
        relative_data_folder + 'nki_rs_en/subdat_2_nkiRSen_resting_harvardoxfordfull_111ROIs.mat',
        relative_data_folder + 'nki_rs_en/subdat_3_nkiRSen_resting_harvardoxfordfull_111ROIs.mat',
        relative_data_folder + 'nki_rs_en/subdat_3_nkiRSen_resting_harvardoxfordfull_111ROIs_MBcleaned.mat',
        relative_data_folder + 'nki_rs_en/subdat_4_nkiRSen_resting_harvardoxfordfull_111ROIs.mat',
    ]
    valid_data = []
    for i, file in enumerate(valid_data_filenames):
        f = h5py.File(file)
        for k, v in f.items():
            valid_data.append(np.array(v))
    valid_data = np.concatenate(valid_data)
    valid_data = np.reshape(valid_data, (-1, 1, CONFIG.MATR_SIZE, CONFIG.MATR_SIZE,))

    # valid labels
    valid_ids = []
    valid_regr_labels = []
    valid_class_labels = []

    valid_labels_filenames = [
        relative_data_folder + 'nki_rs_en/subdat_1_nkiRSen_labels.csv',
        relative_data_folder + 'nki_rs_en/subdat_2_nkiRSen_labels.csv',
        relative_data_folder + 'nki_rs_en/subdat_3_nkiRSen_labels.csv',
        relative_data_folder + 'nki_rs_en/subdat_3_nkiRSen_labels_MBcleaned.csv',
        relative_data_folder + 'nki_rs_en/subdat_4_nkiRSen_labels.csv',
    ]

    for i, file in enumerate(valid_labels_filenames):
        with open(file, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            for currentline in reader:
                valid_ids.append(int(currentline[0]))
                valid_regr_labels.append(int(currentline[2]))
                valid_class_labels.append(int(currentline[1]))
    valid_ids = np.array(valid_ids)
    valid_regr_labels = np.array(valid_regr_labels)
    valid_class_labels = np.array(valid_class_labels)

    # train data
    # currently only the regression dataset!
    train_data_filename = [
        relative_data_folder + 'lmu_sald_combined/CORR_tensor_public_regr.pickle',
        relative_data_folder + 'lmu_sald_combined/CORR_tensor_public.pickle'
    ]
    train_datas = []
    for train_file in train_data_filename:
        with open(train_file, 'rb') as f:
            train_data = pickle.load(f)['data_tensor']
        train_datas.append(np.reshape(train_data, (-1, 1, CONFIG.MATR_SIZE, CONFIG.MATR_SIZE,)))

    # train labels
    train_labels_filename = [
        relative_data_folder + 'lmu_sald_combined/labels_public_regr.csv',
        relative_data_folder + 'lmu_sald_combined/labels_public.csv',
    ]
    train_ids_l = []
    train_regr_labels_l = []
    train_class_labels_l = []
    for train_label_file in train_labels_filename:
        train_ids = []
        train_regr_labels = []
        train_class_labels = []
        with open(train_label_file, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            for currentline in reader:
                train_ids.append(int(currentline[0]))
                train_regr_labels.append(int(currentline[1]))
                train_class_labels.append(int(currentline[2]))
        train_ids = np.array(train_ids)
        train_regr_labels = np.array(train_regr_labels)
        train_class_labels = np.array(train_class_labels)
        train_ids_l.append(train_ids)
        train_regr_labels_l.append(train_regr_labels)
        train_class_labels_l.append(train_class_labels)

    # saving
    np.savez(
        filename,
        train_data_r=train_datas[0].astype(np.float32),
        train_ids_r=train_ids_l[0].astype(np.float32),
        train_regr_labels_r=train_regr_labels_l[0].astype(np.float32),
        train_class_labels_r=train_class_labels_l[0].astype(np.float32),
        train_data_b=train_datas[1].astype(np.float32),
        train_ids_b=train_ids_l[1].astype(np.float32),
        train_regr_labels_b=train_regr_labels_l[1].astype(np.float32),
        train_class_labels_b=train_class_labels_l[1].astype(np.float32),
        valid_data=valid_data.astype(np.float32),
        valid_ids=valid_ids.astype(np.float32),
        valid_regr_labels=valid_regr_labels.astype(np.float32),
        valid_class_labels=valid_class_labels.astype(np.float32),
        test_data=test_data.astype(np.float32),
        test_ids=test_ids.astype(np.float32),
        test_class_labels=test_class_labels.astype(np.float32),
        test_regr_labels=test_regr_labels.astype(np.float32),
    )


def read_datasets(CONFIG):
    relative_data_folder = CONFIG.RELATIVE_DATA_FOLDER
    filename = relative_data_folder + CONFIG.DATASET_PART
    if CONFIG.SUPERVISE_TYPE == 'regress':
        lab_str = '_regr_labels'
        train_str = '_r'
    elif CONFIG.SUPERVISE_TYPE == 'binary':
        lab_str = '_class_labels'
        train_str = '_b'
    else:
        raise NotImplementedError
    with np.load(filename) as df:
        train_data = df['train_data' + train_str]
        train_labels = df['train' + lab_str + train_str]
        valid_data = df['valid_data']
        valid_labels = df['valid' + lab_str]
        test_data = df['test_data']
        test_labels = df['test' + lab_str]
        return {'train': (train_data, train_labels),
                'val': (valid_data, valid_labels),
                'test': (test_data, test_labels)}


def preprocess(df, CONFIG):
    # TODO: finish this
    return df


def prepare_dataset(CONFIG):
    relative_data_folder = CONFIG.RELATIVE_DATA_FOLDER
    filename = relative_data_folder + CONFIG.DATASET_PART
    if not os.path.isfile(filename):
        partition_data(CONFIG)
        print('Data partitioned')
    else:
        print('Data already partitioned')

    # TODO: some preprocess?

    df = read_datasets(CONFIG)
    df = preprocess(df, CONFIG)
    return df
