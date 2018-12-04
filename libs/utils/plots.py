"""
@author István Hajdu at MTA TTK
https://github.com/hajduistvan/connectome_gan
"""
import matplotlib.pyplot as plt
import os
import numpy as np
def load_npz(filename):
    loaded = np.load(filename)
    pure_val = loaded['pure_val']
    pure_dev = loaded['pure_dev']
    mix_val_best_losses = loaded['mix_val']
    mix_dev_best_losses = loaded['mix_dev']
    return pure_val,pure_dev,mix_val_best_losses,mix_dev_best_losses




def plot_learning_curves(train_sizes, val_sizes, pure_val, mix_val_best_losses, pure_dev, mix_dev_best_losses, save_folder=None, saved_file = None):

    total_sizes = [a+b if not a is None and not b is None else 14864+2477 for a,b in zip(train_sizes,val_sizes)]
    pure_val_best_losses = [min(l) for l in pure_val]

    pure_dev_best_losses = [l[np.argmin(ll)] for l, ll in zip(pure_dev, pure_val)]


    # plot 1: val & dev losses
    fig1 = plt.figure()
    pure_val_line, = plt.plot(total_sizes, pure_val_best_losses)
    pure_dev_line, = plt.plot(total_sizes, pure_dev_best_losses)
    mix_val_line, = plt.plot(total_sizes, mix_val_best_losses)
    mix_dev_line, = plt.plot(total_sizes, mix_dev_best_losses)
    # plt.legend((pure_val_line,pure_dev_line,mix_val_line,mix_dev_line),
    #            ('Only original, validation loss', 'Only original, developer set loss',
    #             'Mixed, validation loss','Mixed, developer loss'))
    plt.grid(True)

    plt.show()
    if not save_folder is None:
        os.makedirs(save_folder,exist_ok=True)
        plt.savefig(os.path.join(save_folder, 'val_and_dev_losses2.png'))

    # plot 2 : val & dev differences
    diff_val = [b-a for a,b in zip(mix_val_best_losses,pure_val_best_losses)]
    diff_dev = [b-a for a,b in zip(mix_dev_best_losses,pure_dev_best_losses)]
    fig2 = plt.figure()
    diff_val_line, = plt.plot(total_sizes,diff_val)
    diff_dev_line, = plt.plot(total_sizes,diff_dev)
    plt.legend((diff_val_line,diff_dev_line),
               ('Advantage of mixed training, validation','Advantage of mixed training, dev set'))
    plt.grid(True)
    # plt.show()
    if not save_folder is None:
        os.makedirs(save_folder,exist_ok=True)
        plt.savefig(os.path.join(save_folder, 'difference_val_dev_losses.png'))

def plot_learning_curves_gender(
        train_sizes,
        val_sizes,
        pure_val,
        mix_val_best_losses,
        pure_dev,
        mix_dev_best_losses,
        pure_cnn_val_acces,
        pure_cnn_dev_acces,
        mixed_cnn_val_acces,
        mixed_cnn_dev_acces,
        save_folder=None,
        saved_file = None
):

    total_sizes = [a+b if not a is None and not b is None else 14864+2477 for a,b in zip(train_sizes,val_sizes)]
    pure_val_best_losses = [min(l) for l in pure_val]

    pure_dev_best_losses = [l[np.argmin(ll)] for l, ll in zip(pure_dev, pure_val)]
    pure_val_best_accs = [l[np.argmin(ll)] for l, ll in zip(pure_cnn_val_acces, pure_val)]
    pure_dev_best_accs = [l[np.argmin(ll)] for l, ll in zip(pure_cnn_dev_acces, pure_val)]

    # plot 1: val & dev losses
    fig1 = plt.figure()

    ax = plt.subplot(111)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    pure_val_line, = plt.plot(total_sizes, pure_val_best_losses, label='Orig val loss')
    pure_dev_line, = plt.plot(total_sizes, pure_dev_best_losses, label='Orig dev loss')
    mix_val_line, = plt.plot(total_sizes, mix_val_best_losses, label='Mixed val loss')
    mix_dev_line, = plt.plot(total_sizes, mix_dev_best_losses, label='Mixed dev loss')
    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)

    # plt.show()
    # pure_val_line, = plt.plot(total_sizes, pure_val_best_losses)
    # pure_dev_line, = plt.plot(total_sizes, pure_dev_best_losses)
    # mix_val_line, = plt.plot(total_sizes, mix_val_best_losses)
    # mix_dev_line, = plt.plot(total_sizes, mix_dev_best_losses)
    # plt.legend((pure_val_line,pure_dev_line,mix_val_line,mix_dev_line),
    #            ('Orig val loss', 'Orig dev loss',
    #             'Mixed val loss', 'Mixed dev loss'))
    plt.grid(True)

    plt.show()
    if not save_folder is None:
        os.makedirs(save_folder,exist_ok=True)
        plt.savefig(os.path.join(save_folder, 'val_and_dev_losses.png'))

    # plot 2: val & dev accs
    fig2 = plt.figure()

    ax = plt.subplot(111)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    pure_val_line, = plt.plot(total_sizes, pure_val_best_accs, label='Orig val acc')
    pure_dev_line, = plt.plot(total_sizes, pure_dev_best_accs, label='Orig dev acc')
    mix_val_line, = plt.plot(total_sizes, mixed_cnn_val_acces, label='Mixed val acc')
    mix_dev_line, = plt.plot(total_sizes, mixed_cnn_dev_acces, label='Mixed dev acc')
    # plt.legend((pure_val_line,pure_dev_line,mix_val_line,mix_dev_line),
    #            ('Orig val acc', 'Orig dev acc',
    #             'Mixed val acc','Mixed dev acc'))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    plt.grid(True)

    plt.show()
    if not save_folder is None:
        os.makedirs(save_folder,exist_ok=True)
        plt.savefig(os.path.join(save_folder, 'val_and_dev_acces.png'))

    #
    # # plot 3 : val & dev loss differences
    # diff_val = [b-a for a,b in zip(mix_val_best_losses,pure_val_best_losses)]
    # diff_dev = [b-a for a,b in zip(mix_dev_best_losses,pure_dev_best_losses)]
    # fig2 = plt.figure()
    # diff_val_line, = plt.plot(total_sizes,diff_val)
    # diff_dev_line, = plt.plot(total_sizes,diff_dev)
    # plt.legend((diff_val_line,diff_dev_line),
    #            ('Advantage of mixed training, validation','Advantage of mixed training, dev set'))
    # plt.grid(True)
    # # plt.show()
    # if not save_folder is None:
    #     os.makedirs(save_folder,exist_ok=True)
    #     plt.savefig(os.path.join(save_folder, 'difference_val_dev_losses.png'))
