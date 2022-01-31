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

2. Create a conda environment using the `environment.yml` file, to install the dependencies:  
`$ conda env create -f environment.yml`

3. Activate the new conda environment:
`$ conda activate deepfashion`

### Getting the data

1. You can download the dataset from the Google Drive here:
https://drive.google.com/drive/folders/0B7EVK8r0v71pWGplNFhjc01NbzQ

2. Download the training data into some directory which is appropriate. 
   The Train data are located in the following folders:
   - Teeth-Segmentation/SemiSupervised/Selfsupervised Clustered Train Data/
   - Teeth-Segmentation/SemiSupervised/Supervised Labeled Train Data/
 ## Running experiments
3. Run step1_data_augmentation.py to generate augmented data
4. Run step1_data_augmentation.py to generate train and valid lists
### Training the models

5. Run step3_trainingSSKNN.py to train the model.

   
### Testing the models
6. Run step4_test.py to test the model on the test data located in:
   Teeth-Segmentation/Labeled Test Data/
   
We provide pre-trained supervised and semisupervised segmentation models as well as the results for testing the two models on the test data in the directory:
Teeth-Segmentation/SemiSupervised/models/

Achnowledgement:
The code is inspired by MeshSegNet https://github.com/Tai-Hsien/MeshSegNet and 
PointCloudLearningACD https://github.com/matheusgadelha/PointCloudLearningACD
Use one of the two provided requirements files. 
