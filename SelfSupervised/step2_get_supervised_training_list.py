import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd
import random

if __name__ == '__main__':

    seed_value = 1234
    random.seed(seed_value)
    np.random.seed(seed_value)
    data_path = 'C:/Users/Poly/OneDrive - polymtl.ca/Documents - DentalAI/Données/Data to publish/Experiment/Code/SelfSupervised/Supervised Train registered_vtp'
    outputPath = './Supervised Train registered_vtp'

    num_augmentations = 20
    train_size = 0.8
    with_flip = True

    # List contains all the arches with their augmentation
    mesh_list = [i for i in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, i)) and 'vtp' in i]  # and 'Lower' in i
    # List containing unique cases without any augmentation
    sample_list = [i.split('_')[0] for i in mesh_list if 'A' not in i and 'reflected' not in i]

    sample_name = 'A{0}_Sample_0{1}_d.vtp'

    # remove duplicated
    sample_list = list(dict.fromkeys(sample_list))
    sample_list = np.asarray(sample_list)

    i_cv = 0
    kf = KFold(n_splits=3, shuffle=True, random_state=seed_value)
    for train_idx, val_list in kf.split(sample_list):
        i_cv += 1
        print('Round:', i_cv)

        test_list = []# now we are not creating a test set
        train_list, val_list = sample_list[train_idx], sample_list[val_list]

        print('Training list:\n', train_list, '\nValidation list:\n', val_list, '\nTest list:\n', test_list)

        #training
        train_name_list = []
        for i_sample in train_list:
            for arch in mesh_list:
                if i_sample in arch:
                    train_name_list.append(outputPath + '/'+arch)

        with open(os.path.join(data_path, 'train_list_{0}.csv'.format(i_cv)), 'w') as file:
            for f in train_name_list:
                file.write(f+'\n')

        #validation
        val_name_list = []
        for i_sample in val_list:
            for arch in mesh_list:
                if i_sample in arch:
                    val_name_list.append(outputPath +'/' +arch)

        with open(os.path.join(data_path, 'val_list_{0}.csv'.format(i_cv)), 'w') as file:
            for f in val_name_list:
                file.write(f+'\n')

        print('--------------------------------------------')
        print('with flipped samples:', with_flip)
        print('# of train:', len(train_name_list))
        print('# of validation:', len(val_name_list))
        print('--------------------------------------------')