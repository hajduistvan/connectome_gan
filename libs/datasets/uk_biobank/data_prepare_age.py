"""
@author István Hajdu at MTA TTK
https://github.com/hajduistvan/connectome_gan
"""
import os
import csv
import matplotlib.pyplot as plt
data_folder = 'F:/HajduIstvan/UK_Biobank/'
import numpy as np
import torch
from torch.utils import data
from ast import literal_eval as make_tuple

class UKBioBankPreprocesser:
    def __init__(self, data_folder, matrix_size=55):
        self.data_folder = data_folder
        self.matrix_size = matrix_size

    def read_full_csv(self):
        filename = os.path.join(self.data_folder, 'ukb24364.csv')
        with open(filename) as f:
            reader = csv.reader(f)
            csv_dict = {}
            for i, currentline in enumerate(reader):
                if i == 0:
                    keys = currentline.copy()
                    for k in keys:
                        csv_dict[k] = []
                else:
                    for j, v in enumerate(currentline):
                        csv_dict[keys[j]].append(v)
        return csv_dict

    def convert_dates_to_age(self, year_of_birth, month_of_birth, date_of_imaging):
        year_of_birth = int(year_of_birth)
        month_of_birth = int(month_of_birth)
        year_of_imaging = int(date_of_imaging[:4])
        month_of_imaging = int(date_of_imaging[5:7])
        age = year_of_imaging - year_of_birth + (month_of_imaging - month_of_birth) / 12
        return age

    def get_ids_ages_and_save(self, csv_dict):
        filename = os.path.join(self.data_folder, "biobank_eids_ages.csv")
        year_of_birth_key = '34-0.0'
        month_of_birt_key = '52-0.0'
        date_of_imaging_key = '53-2.0'
        eid_key = 'eid'
        eids, ages = [], []
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            for i, eid in enumerate(csv_dict[eid_key]):
                age = self.convert_dates_to_age(csv_dict[year_of_birth_key][i],
                                                csv_dict[month_of_birt_key][i],
                                                csv_dict[date_of_imaging_key][i])
                eid = int(eid)
                eids.append(eid)
                ages.append(age)
                writer.writerow([eid, age])
        return eids, ages

    def read_matrix(self, filename, z_score_format=False):
        matrix = np.ones((self.matrix_size, self.matrix_size)).astype(np.float32)
        if z_score_format:
            matrix *= np.inf
        i, j = 0, 1
        with open(filename, 'r') as f:
            line = f.readlines()[0]
            elements = line.split()
            # print('len_of_lines: ', len(elements))
            for idx, element in enumerate(elements):
                if z_score_format:
                    matrix[i, j] = matrix[j, i] = float(element)
                else:
                    matrix[i, j] = matrix[j, i] = np.tanh(float(element))
                if j == self.matrix_size - 1:
                    i += 1
                    j = i + 1
                else:
                    j += 1
        return matrix

    def read_all_matrices(self):
        eids = []
        matrices = []
        root_elements = os.listdir(self.data_folder)
        for element in root_elements:
            element = os.path.join(self.data_folder, element)
            if os.path.isdir(element):
                # print(element)
                filenames_in_subfolder = os.listdir(element)
                # print(filenames_in_subfolder)
                for filename in filenames_in_subfolder:
                    if os.path.splitext(filename)[1] == '.txt':
                        eids.append(int(os.path.basename(filename).split('_')[0]))
                        matrices.append(self.read_matrix(os.path.join(element, filename)))
        return eids, matrices

    def partition_indeces(self, eids, ratios=(0.75, 0.875)):
        idx_eids = np.random.permutation(len(eids))
        idx_0 = int(np.around(len(eids) * ratios[0]))
        idx_1 = int(np.around(len(eids) * ratios[1]))
        # train, val, test
        return idx_eids[:idx_0], idx_eids[idx_0:idx_1], idx_eids[idx_1:]

    def save_partitioned_dataset_to_npz(self, eids_label, ages, eids_matrices, matrices, ratios=(0.75, 0.875)):
        filename = os.path.join(self.data_folder, 'partitioned_dataset.npz')
        if not os.path.isfile(os.path.join(data_folder, filename)):
            partition_indeces = self.partition_indeces(eids_matrices, ratios)
            # print(len(partition_indeces[0]))
            # print(len(partition_indeces[1]))
            # print(len(partition_indeces[2]))
            dataset_list = []

            for idx_partition in partition_indeces:
                dataset = []
                for i, idx in enumerate(idx_partition):
                    if eids_matrices[idx] in eids_label:
                        dataset.append((matrices[idx], ages[eids_label.index(eids_matrices[idx])], eids_matrices[idx]))
                dataset_list.append(dataset)
            np.savez(
                filename,
                train_dataset=dataset_list[0],
                val_dataset=dataset_list[1],
                test_dataset=dataset_list[2]
            )
            print('Partitioned data saved. Train: ' + str(len(partition_indeces[0])) + '. Val: ' + str(
                len(partition_indeces[1])) \
                  + '. Test: ' + str(len(partition_indeces[2])))
        else:
            print('Data already partitioned')

    def visualize_data(self, ids):
        vis_folder = os.path.join(self.data_folder, 'vis_images')
        os.makedirs(vis_folder, exist_ok=True)
        filename = os.path.join(self.data_folder, 'partitioned_dataset.npz')
        with np.load(filename) as df:
            train_dataset = df['train_dataset']
            val_dataset = df['val_dataset']
            test_dataset = df['test_dataset']
        joined_dataset = np.concatenate((train_dataset, val_dataset, test_dataset), 0)
        for i in ids:
            plt.imshow(joined_dataset[i,0], vmin=-1, vmax=1, cmap='jet', interpolation='nearest')
            plt.title('EID: '+str(joined_dataset[i,2])+'. Age: '+str(joined_dataset[i,1]))
            plt.colorbar()
            plt.savefig(os.path.join(vis_folder, str(joined_dataset[i,2])+'.png'))
            plt.show()
    def get_min_max_age(self):
        m, M = 1000, 0
        filename = os.path.join(self.data_folder, 'partitioned_dataset.npz')
        with np.load(filename) as df:
            train_dataset = df['train_dataset']
            val_dataset = df['val_dataset']
            test_dataset = df['test_dataset']
        joined_dataset = np.concatenate((train_dataset, val_dataset, test_dataset), 0)
        for i in range(joined_dataset.shape[0]):
            if joined_dataset[i,1] < m:
                m = joined_dataset[i,1]
            if joined_dataset[i,1] > M:
                M = joined_dataset[i,1]
        return m,M
