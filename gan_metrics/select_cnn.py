import csv
import os
import torch
from models.classifier import ConnectomeConvInferenceNet


def concat_csvs(exp_names):
    run_dir = '/home/orthopred/repositories/conn_gan/cnn_arch_search/runs'
    concat_list = []
    for exp_name in exp_names:
        csv_filename = os.path.join(run_dir, exp_name, 'cnn_search_result.csv')
        with open(csv_filename, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                concat_list.append([*row, exp_name])
    with open(os.path.join(run_dir, exp_names[-1], "cnn_search_final_result.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerows(concat_list)
    return concat_list

def get_model(target_gpu_id, concat_list_idx, row=None):
    exp_names = ['gpu0_run0', 'gpu0_run1', 'gpu1_run0', 'gpu1_run1', 'gpu1_run2']
    concat_list = concat_csvs(exp_names)
    # row_header: [gpu_id, step, loss, acc, num_params, c1_w, c2_w, lr, mom, wd, exp_name]
    run_dir = 'cnn_arch_search/runs'
    if row is None:
        csv_row = concat_list[concat_list_idx-1]
        gpu_id, step, loss, acc, num_params, c1_w, c2_w, lr, mom, wd, exp_name = csv_row
    else:
        gpu_id, step, loss, acc, num_params, c1_w, c2_w, lr, mom, wd, exp_name = row
    ckpt_file = os.path.join('/home/orthopred/repositories/conn_gan/', run_dir, exp_name, gpu_id + step + '.pth')
    state_dict = torch.load(ckpt_file)['state_dict']
    # print(c1_w, c2_w)
    model = ConnectomeConvInferenceNet([int(c1_w), int(c2_w)], state_dict, target_gpu_id)
    return model
