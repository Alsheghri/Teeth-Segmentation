'
MeshSegNet https://github.com/Tai-Hsien/MeshSegNet an

from torch.utils.data import Dataset
import pandas as pd
import torch
import random
import numpy as np
from vedo import *
from scipy.spatial import distance_matrix
from easy_mesh_vtk import *

class Mesh_Dataset(Dataset):
    def __init__(self, data_list_path, num_classes=15, patch_size=7000):
        """
        Args:
            h5_path (string): Path to the txt file with h5 files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_list = pd.read_csv(data_list_path, header=None)
        self.num_classes = num_classes
        self.patch_size = patch_size

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i_mesh = self.data_list.iloc[idx][0] #vtk file name
        print('Supervised_Mesh=', i_mesh)
        # read vtk
        mesh = load(i_mesh)
        labels = mesh.getCellArray('Label').astype('int32').reshape(-1, 1)

        # move mesh to origin
        cells = np.zeros([mesh.NCells(), 9], dtype='float32')
        for i in range(len(cells)):
            cells[i][0], cells[i][1], cells[i][2] = mesh._data.GetPoint(mesh._data.GetCell(i).GetPointId(0)) # don't need to copy
            cells[i][3], cells[i][4], cells[i][5] = mesh._data.GetPoint(mesh._data.GetCell(i).GetPointId(1)) # don't need to copy
            cells[i][6], cells[i][7], cells[i][8] = mesh._data.GetPoint(mesh._data.GetCell(i).GetPointId(2)) # don't need to copy

        mean_cell_centers = mesh.centerOfMass()
        cells[:, 0:3] -= mean_cell_centers[0:3]
        cells[:, 3:6] -= mean_cell_centers[0:3]
        cells[:, 6:9] -= mean_cell_centers[0:3]

        ####################################
        # output to visualize
        mesh0 = Easy_Mesh()
        mesh0.cells = cells
        mesh0.update_cell_ids_and_points()
        #mesh0.to_obj('tmp0.obj')

        # customized normal calculation; the vtk/vedo build-in function will change number of points
        v1 = np.zeros([mesh.NCells(), 3], dtype='float32')
        v2 = np.zeros([mesh.NCells(), 3], dtype='float32')
        v1[:, 0] = cells[:, 0] - cells[:, 3]
        v1[:, 1] = cells[:, 1] - cells[:, 4]
        v1[:, 2] = cells[:, 2] - cells[:, 5]
        v2[:, 0] = cells[:, 3] - cells[:, 6]
        v2[:, 1] = cells[:, 4] - cells[:, 7]
        v2[:, 2] = cells[:, 5] - cells[:, 8]
        mesh_normals = np.cross(v1, v2)
        mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
        mesh_normals[:, 0] /= mesh_normal_length[:]
        mesh_normals[:, 1] /= mesh_normal_length[:]
        mesh_normals[:, 2] /= mesh_normal_length[:]
        mesh.addCellArray(mesh_normals, 'Normal')

        # prepre input and make copies of original data
        points = mesh.points().copy()
        points[:, 0:3] -= mean_cell_centers[0:3]
        normals = mesh.getCellArray('Normal').copy() # need to copy, they use the same memory address
        barycenters = mesh.cellCenters() # don't need to copy
        barycenters -= mean_cell_centers[0:3]

        #normalized data
        maxs = points.max(axis=0)
        mins = points.min(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)

        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i]     #point 1
            cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
            cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
            barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
            normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]

        X = np.column_stack((cells, barycenters, normals))
        Y = labels

        # initialize batch of input and label
        X_train = np.zeros([self.patch_size, X.shape[1]], dtype='float32')
        Y_train = np.zeros([self.patch_size, Y.shape[1]], dtype='int32')
        S1 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
        S2 = np.zeros([self.patch_size, self.patch_size], dtype='float32')

        # calculate number of valid cells (tooth instead of gingiva)
        positive_idx = np.argwhere(labels>0)[:, 0] #tooth idx
        negative_idx = np.argwhere(labels==0)[:, 0] # gingiva idx

        num_positive = len(positive_idx) # number of selected tooth cells
        if num_positive > self.patch_size: # all positive_idx in this patch
            positive_selected_idx = np.random.choice(positive_idx, size=self.patch_size, replace=False)
            selected_idx = positive_selected_idx
        else:   # patch contains all positive_idx and some negative_idx
            num_negative = self.patch_size - num_positive # number of selected gingiva cells
            positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
            negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
            selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))

        selected_idx = np.sort(selected_idx, axis=None)

        X_train[:] = X[selected_idx, :]
        Y_train[:] = Y[selected_idx, :]

        D = distance_matrix(X_train[:, 9:12], X_train[:, 9:12])
        S1[D<0.1] = 1.0
        S1 = S1 / np.dot(np.sum(S1, axis=1, keepdims=True), np.ones((1, self.patch_size)))

        S2[D<0.2] = 1.0
        S2 = S2 / np.dot(np.sum(S2, axis=1, keepdims=True), np.ones((1, self.patch_size)))

        X_train = X_train.transpose(1, 0)
        Y_train = Y_train.transpose(1, 0)

        sample = {'cells': torch.from_numpy(X_train), 'labels': torch.from_numpy(Y_train),
                  'A_S': torch.from_numpy(S1), 'A_L': torch.from_numpy(S2)}

        return sample


class KNNSelfSupDataset(Dataset):
    def __init__(self, root='/home/ammar15/scratch/data/perfectUpperArches30/clustered vtp files/',
                 num_classes=15, patch_size=7000,
                 npoints=2500,
                 class_choice=None,
                 normal_channel=False,
                 k_shot=-1,
                 use_val=False,
                 exclude_fns=[]):
        '''
            Expected self-supervised dataset folder structure:

                ROOT
                  |--- <sub-folder-1>
                  |     | -- af55f398af2373aa18b14db3b83de9ff.npy
                  |     | -- ff77ea82fb4a5f92da9afa637af35064.npy
                  |    ...
                  |
                  |--- <sub-folder-2>
                 ...

            The "subfolders" loosely correspond to "object categories", but can
            be arbitrary. The code works with a single subfolder. However, it
            does not work if there are no subfolders at all under the ROOT path.

        '''
        self.npoints = npoints
        self.root = root
        self.normal_channel = normal_channel
        self.k_shot = k_shot
        self.meta = {}
        subfolders = os.listdir(root)
        self.classes_original = dict(zip(subfolders, range(len(subfolders))))
        self.cat = self.classes_original
        self.use_val = use_val
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.fns = [f for f in os.listdir(self.root) if f.endswith('.vtp')]
        NUM_SAMPLES =len(self.fns)
        # support for specifying a random subset of the self-sup data
        if self.k_shot > 0:
            print('Subsampling self-supervised dataset (%d samples).' % self.k_shot)
            #self.fns = random.sample(self.fns, self.k_shot)

        if self.use_val:
            # we fix 80/20 train/val splits per category
            self.fns = random.sample(self.fns, math.floor(NUM_SAMPLES * 0.8))

    def __len__(self):
        self.fns = [f for f in os.listdir(self.root) if f.endswith('.vtp')]
        return len(self.fns)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
      
        i_mesh = self.fns[idx]  # vtk file name
        print('SelfSupervised_Mesh=', i_mesh)
        # read vtk
        vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-10, 10], rotate_Y=[-10, 10], rotate_Z=[-10, 10],
                                                translate_X=[-1, 1], translate_Y=[-1, 1], translate_Z=[-1, 1],
                                                scale_X=[0.9, 1.1], scale_Y=[0.9, 1.1],
                                                scale_Z=[0.9, 1.1])  # use default random setting

        mesh = Easy_Mesh(os.path.join(self.root, i_mesh))
        mesh.mesh_transform(vtk_matrix)

        labels = mesh.cell_attributes['Label'].astype('int32')

        cell_centers = (mesh.cells[:, 0:3] + mesh.cells[:, 3:6] + mesh.cells[:, 6:9]) / 3.0
        mean_cell_centers = np.mean(cell_centers, axis=0)
        mesh.cells[:, 0:3] -= mean_cell_centers[0:3]
        mesh.cells[:, 3:6] -= mean_cell_centers[0:3]
        mesh.cells[:, 6:9] -= mean_cell_centers[0:3]
        mesh.update_cell_ids_and_points()  # update object when change cells
        mesh.get_cell_normals()  # get cell normal
        cells = mesh.cells[:]
        normals = mesh.cell_attributes['Normal'][:]
        cell_ids = mesh.cell_ids[:]
        points = mesh.points[:]

        barycenters = (cells[:, 0:3] + cells[:, 3:6] + cells[:, 6:9]) / 3
        # normalized data
        maxs = points.max(axis=0)
        mins = points.min(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)

        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i]  # point 1
            cells[:, i + 3] = (cells[:, i + 3] - means[i]) / stds[i]  # point 2
            cells[:, i + 6] = (cells[:, i + 6] - means[i]) / stds[i]  # point 3
            barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
            normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

        X = np.column_stack((cells, barycenters, normals))
        Y = labels
        positive_idx = np.argwhere(labels >= 0)[:, 0]  # tooth idx
        # initialize batch of input and label
        X_train = np.zeros([self.patch_size, X.shape[1]], dtype='float32')
        Y_train = np.zeros([self.patch_size, Y.shape[1]], dtype='int32')
        S1 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
        S2 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
        # calculate number of valid cells (tooth instead of gingiva)
        positive_idx = np.argwhere(labels >= 0)[:, 0]  # tooth idx
        
        num_positive = len(positive_idx)  # number of selected tooth cells
        positive_selected_idx = np.random.choice(positive_idx, size=self.patch_size, replace=False)
        try:
          selected_idx = np.sort(positive_selected_idx, axis=None)
        except:
          print(i_mesh) 

        X_train[:] = X[selected_idx, :]
        Y_train[:] = Y[selected_idx, :]

        D = distance_matrix(X_train[:, 9:12], X_train[:, 9:12])
        S1[D < 0.1] = 1.0
        S1 = S1 / np.dot(np.sum(S1, axis=1, keepdims=True), np.ones((1, self.patch_size)))

        S2[D < 0.2] = 1.0
        S2 = S2 / np.dot(np.sum(S2, axis=1, keepdims=True), np.ones((1, self.patch_size)))

        X_train = X_train.transpose(1, 0)
        Y_train = Y_train.transpose(1, 0)

        sample = {'cells': torch.from_numpy(X_train), 'labels': torch.from_numpy(Y_train),
                  'A_S': torch.from_numpy(S1), 'A_L': torch.from_numpy(S2)}

        return sample

if __name__ == '__main__':
    KNN_ROOT='C:/Users/Poly/PycharmProjects/MeshSegNet/30perfectUpperArches/clustered vtp files/'
    labeled_fns = []
    SELFSUP_DATASET = KNNSelfSupDataset(root=KNN_ROOT, num_classes=15, patch_size=7000,
                                        npoints=200,
                                        class_choice=None,
                                        normal_channel=False,
                                        k_shot=13,
                                        use_val=False,
                                        exclude_fns=labeled_fns)



    selfsupDataLoader = torch.utils.data.DataLoader(SELFSUP_DATASET,
                                                    batch_size=2,
                                                    shuffle=True, num_workers=4)


    selfsupIterator = next(iter(selfsupDataLoader))
    sample = selfsupIterator
    print(selfsupIterator)
