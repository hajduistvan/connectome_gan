import numpy as np
import os
import argparse

def create_dummy_dataset(dir):
    os.makedirs(dir, exist_ok=True)
    split_sizes = [14864, 2477, 2477]
    labels = [np.random.randint(0, 2, (s,)) for s in split_sizes]

    datasets = []

    for i in range(3):
        dataset = []
        matrices = np.random.rand(split_sizes[i], 55, 55)*2-1
        matrices[labels[i] == 1] = 0.8 * matrices[labels[i]==1] ** 3 + 0.2*(np.random.rand(*(matrices[labels[i]==1]).shape)*2-1)
        matrices = (matrices + np.transpose(matrices, [0, 2, 1])) / 2
        for j in range(55):
            matrices[:, j, j] = 1

        for j in range(split_sizes[i]):
            dataset.append((matrices[j], labels[i][j]))
        datasets.append(dataset)


    data = {
        'train_dataset': datasets[0],
        'val_dataset': datasets[1],
        'test_dataset': datasets[2]
    }
    np.savez(os.path.join(dir, 'partitioned_dataset_gender.npz'), **data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=os.path.join(os.getcwd(), 'dummy_dataset/'))
    args = parser.parse_args()
    create_dummy_dataset(args.dir)

