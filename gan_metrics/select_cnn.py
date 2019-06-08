import csv
import os
import torch
from models.classifier import ConnectomeConvInferenceNet


def concat_csvs(run_dir):
    """
    As the results of CNN hyperparameter searches can be in multiple csvs, we have to concatenate them first.
    :param run_dir: 'run_dir' argument that was given to the cnn_search.py runs.
    :return:
    """
    concat_list = []
    exp_names = os.listdir(run_dir)  #

    try:
        with open(os.path.join(run_dir, "cnn_search_final_result.csv"), mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                concat_list.append(row)
    except:
        for exp_name in exp_names:
            csv_filename = os.path.join(run_dir, exp_name, 'cnn_search_result.csv')
            try:
                with open(csv_filename, mode='r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    for row in csv_reader:
                        concat_list.append([*row, exp_name])
            except:
                print('Warning: no .csv file at ' + csv_filename + '.')
        with open(os.path.join(run_dir, "cnn_search_final_result.csv"), "w") as f:
            writer = csv.writer(f)
            writer.writerows(concat_list)
    return concat_list


def get_model(target_gpu_id, concat_list_idx, run_dir, row=None):
    """
    Loads the desired model, e.g. the Reference Net.
    :param target_gpu_id:
    :param concat_list_idx: row id of the model in the concatenated csv file
    :param run_dir: jsut like above.
    :param row: as an overwriter, the content of the corresponding row of the csv file can be given.
    :return: the loaded CNN model.
    """
    concat_list = concat_csvs(run_dir)
    # row_header: [gpu_id, step, loss, acc, num_params, c1_w, c2_w, lr, mom, wd, exp_name]
    if row is None:
        csv_row = concat_list[concat_list_idx - 1]
        gpu_id, step, loss, acc, num_params, c1_w, c2_w, lr, mom, wd, exp_name = csv_row
    else:
        gpu_id, step, loss, acc, num_params, c1_w, c2_w, lr, mom, wd, exp_name = row
    # print(gpu_id, step, loss, acc, num_params, c1_w, c2_w, lr, mom, wd, exp_name)
    ckpt_file = os.path.join(run_dir, exp_name, gpu_id + step + '.pth')
    state_dict = torch.load(ckpt_file)['state_dict']
    model = ConnectomeConvInferenceNet([int(c1_w), int(c2_w)], state_dict, target_gpu_id)
    return model


if __name__== "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_dir", type=str, default=os.path.join(os.getcwd(), 'cnn_arch_search/runs'))
    args = parser.parse_args()
    _=concat_csvs(args.run_dir)
