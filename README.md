# Tooth Segmentation
## Tooth Segmentation
### ToothSegmentation
<img width="750" alt="Screen Shot 2022-01-06 at 1 26 20 PM" src="https://user-images.githubusercontent.com/6019935/150260947-4d8a8601-5cc0-4e6d-8890-d1ae2d5bae98.png">
## Setting up the project

This repository is still under construction ... 

### Cloning the repository:
### Environment setup

1. Install Anaconda, if not already done, by following these instructions:
https://docs.anaconda.com/anaconda/install/linux/  

2. Create a conda environment using the `environment.yaml` file, to install the dependencies:  
`$ conda env create -f environment.yaml`

3. Activate the new conda environment:
`$ conda activate deepfashion`
### Getting the data

1. You can download the dataset from the Google Drive here:
https://drive.google.com/drive/folders/0B7EVK8r0v71pWGplNFhjc01NbzQ

2. Extract the dataset into some directory which is appropriate
 ## Running experiments

### Training the models

The script used to train a model is `train.py`. Here is the script we used to train our *best* model:
### Testing the models

Achnowledgement:
Our code is inspired by MeshSegNet https://github.com/Tai-Hsien/MeshSegNet and 
PointCloudLearningACD https://github.com/matheusgadelha/PointCloudLearningACD
use the requirements files 
