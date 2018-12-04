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

class UKBioBankGenderPreprocesser:
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



    def get_ids_genders_and_save(self, csv_dict):
        filename = os.path.join(self.data_folder, "biobank_eids_genders.csv")
        gender_key = '31-0.0'
        eid_key = 'eid'
        eids, genders = [], []
        with open(filename, 'w') as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            for i, eid in enumerate(csv_dict[eid_key]):
                gender = int(csv_dict[gender_key][i])
                eid = int(eid)
                eids.append(eid)
                genders.append(gender)
                writer.writerow([eid, gender])
        return eids, genders





    def get_matrices_eids_from_saved_file(self, age_filename):
        with np.load(age_filename) as df:
            train_dataset = df['train_dataset']
            print('train_dataset_shape: ', train_dataset.shape)
            val_dataset = df['val_dataset']
            test_dataset = df['test_dataset']
            return (train_dataset[:,0], val_dataset[:,0], test_dataset[:,0]),\
                   (train_dataset[:,2], val_dataset[:,2], test_dataset[:,2])


    def save_partitioned_dataset_to_npz(self, eids_label, genders):
        filename = os.path.join(self.data_folder, 'partitioned_dataset_gender.npz')
        age_filename = os.path.join(self.data_folder, 'partitioned_dataset.npz')
        if not os.path.isfile(os.path.join(data_folder, filename)):
            partition_matrices, partition_eids = self.get_matrices_eids_from_saved_file(age_filename)
            dataset_list = []

            for matrices_partition,eids_partition in zip(partition_matrices,partition_eids):
                dataset = []
                for i, eid in enumerate(eids_partition):
                    dataset.append((matrices_partition[i], genders[eids_label.index(eid)], eid))
                dataset_list.append(dataset)
            np.savez(
                filename,
                train_dataset=dataset_list[0],
                val_dataset=dataset_list[1],
                test_dataset=dataset_list[2]
            )


    def visualize_data(self, ids):
        vis_folder = os.path.join(self.data_folder, 'vis_images_gender')
        os.makedirs(vis_folder, exist_ok=True)
        filename = os.path.join(self.data_folder, 'partitioned_dataset_gender.npz')
        with np.load(filename) as df:
            train_dataset = df['train_dataset']
            val_dataset = df['val_dataset']
            test_dataset = df['test_dataset']
        joined_dataset = np.concatenate((train_dataset, val_dataset, test_dataset), 0)
        for i in ids:
            plt.imshow(joined_dataset[i,0], vmin=-1, vmax=1, cmap='jet', interpolation='nearest')
            plt.title('EID: '+str(joined_dataset[i,2])+'. Sex: '+['Female', 'Male'][joined_dataset[i,1]])
            plt.colorbar()
            plt.savefig(os.path.join(vis_folder, str(joined_dataset[i,2])+'.png'))
            plt.show()

