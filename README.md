# Tooth Segmentation
## Tooth Segmentation
Created by Ammar Alsheghri, Farnoosh Ghadiri, Ying Zhang, Olivier Lessard, Julia Keren, Farida Cheriet, Francois Guibault

[arXiv](https://arxiv.org/abs/2208.05539]) [Dataset]([https://drive.google.com/drive/folders/1YRCakTTTyr8Sqp3pBBqvgMgZKb59ktES]) [Models]

This repository contains PyTorch implementation for Semi-supervised segmentation of tooth from 3D Scanned Dental Arches (SPIE 2022).

Dental offices tackle thousands of dental reconstructions every year. Complexity and abnormalities in dentition make segmentation of an optical scan a challenging manual task that takes 45 minutes on average. The present work improves the generalization of currently available deep learning segmentation model on 3D dental arches by introducing a new loss function to leverage unlabeled available data. The semi-supervised segmentation network is trained using a joint loss that combines a supervised loss of annotated input and a self-supervised loss of non-labeled input. Our results showed that combining self-supervised and supervised learning improved the segmentation score by 13 % compared with purely supervised learning for the same amount of labeled data. It is concluded that combining representations obtained from self-supervised learning with supervised learning improves the generalization of the 3D tooth segmentation model in the case of few available labeled data.
### ToothSegmentation
<img width="750" alt="Screen Shot 2022-01-06 at 1 26 20 PM" src="https://user-images.githubusercontent.com/6019935/150260947-4d8a8601-5cc0-4e6d-8890-d1ae2d5bae98.png">
## Setting up the project

This repository is still under construction ... 

### Environment setup

1. Install Anaconda, if not already done, by following these instructions:
https://docs.anaconda.com/anaconda/install/linux/  

2. Create a conda environment using the `environment.yml` file, to install the dependencies:  
`$ conda env create -f environment.yml`

Otherwise you can use the file requirements_pip.txt to install the dependencies using pip. 

3. Activate the new conda environment:
`$ conda activate TeethSeg`

### Getting the data

Download the training data into some directory which is appropriate. 
The Train data are located in the following folders:
   - Teeth-Segmentation/SemiSupervised/Selfsupervised Clustered Train Data/
   - Teeth-Segmentation/SemiSupervised/Supervised Labeled Train Data/
 ## Running experiments
1. Run step1_data_augmentation.py to generate augmented data
2. Run step2_get_supervised_training_list.py to generate train and valid lists
### Training the models

3. Run step3_trainingSSKNN.py to train the model.
   
### Testing the models
4. Run step4_test.py to test the model on the test data located in:
   Teeth-Segmentation/SemiSupervised/Labeled Test Data/
   
We provide pre-trained supervised and semisupervised segmentation models as well as the results for testing the two models on the test data in the directory:
Teeth-Segmentation/SemiSupervised/models/

Achnowledgement:

The code is inspired by MeshSegNet https://github.com/Tai-Hsien/MeshSegNet and 
PointCloudLearningACD https://github.com/matheusgadelha/PointCloudLearningACD

### Original Data
Original data with original resolution is available for download from: https://drive.google.com/drive/folders/1YRCakTTTyr8Sqp3pBBqvgMgZKb59ktES?usp=sharing


### Citation
If you find this repository useful, please cite our paper: 
Alsheghri A. A., Ghadiri F., Zhang Y, Lessard O., Keren J., Cheriet F., Guibault F., Semi-supervised segmentation of tooth from 3D Scanned Dental Arches, SPIE medical imaging (Paper No.	12032-101), San Diego, United States, 2022.
