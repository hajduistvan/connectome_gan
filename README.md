# Connectome GAN
MSc thesis repository for training GANs on the UK Biobank resting-state functional connectivity matrices repository. Used packages can be seen in pip_list.txt

## Dataset
Data must be requested from the UK Biobank team. A dummy dataset can be used for test runs.
 
 
 ## Usage
Make sure all scripts are ran with the same working dir. 
Tested on Ubuntu 18.04 with python 3.6. CUDA is nequired.
1. Create dummy dataset with create_dummy_dataset.py
2. Run CNN hyperparameter search with cnn_search.py
3. Create the result csv table with select_cnn.py
4. Choose CNN model id from table (the row index)
5. Train GAN with gan_train.py
 
## Python file descriptions
##### data_handling/dataset.py
Defines the torch Datasets we used. The source of the data is an npz file of numpy arrays, with keys 'train_dataset','val_dataset', and 'test_dataset'.
Each dataset contains an array with shape: [[L, 55, 55], [L]] corresponding to the matrices and the labels. The labels were the gender of the patient, meaning binary classification.

##### models/classifier.py, models/cond_gan.py
Implements the classificator and the gan model.

##### gan_metrics/select_cnn
Used for loading cnn weights that arise from hyperparameter search.
##### gan_metrics/calc_metrics
Used for calculating FID, WAD, and plotting the activations of the Reference Network.

##### cnn_search.py
Used for hyperparameter searching for finding good CNNs.

##### gan_train.py
Training the GANs.

##### test_wad.py & noised_learning.py
Used for testing WAD with noisy data.

##### plot_thesis_imgs.py
Used for some final plots that were needed for the thesis document.

##### cnn_search.yaml
Config file for the CNN hyperparameter search.

##### gan_cfg.yaml
Config file for the GAN training

##### plot_lr_curve_cfg.yaml
Config for testing wad & other plots.

##Contact
Please feel free to give feedback via issues or email: h.istvan95@gmail.com