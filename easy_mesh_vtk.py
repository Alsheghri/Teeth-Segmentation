import sys
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
import math
#import pyvista as pv
#from pymeshfix._meshfix import PyTMesh
#import pymeshfix as mf
#from pymeshfix.examples import planar_mesh
#from pymeshfix import MeshFix
#from pygco import cut_from_graph
import os
from scipy.spatial import distance_matrix
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
#import trimesh


class Easy_Mesh(object):
    def __init__(self, filename=None, warning=False):
        # initialize
        self.warning = warning
        self.reader = None
        self.vtkPolyData = None
        self.cells = np.array([])
        self.cell_ids = np.array([])
        self.points = np.array([])
        self.point_attributes = dict()
        self.cell_attributes = dict()
        self.filename = filename
        if self.filename != None:
            if self.filename[-3:].lower() == 'vtp':
                self.read_vtp(self.filename)
            elif self.filename[-3:].lower() == 'stl':
                self.read_stl(self.filename)
            elif self.filename[-3:].lower() == 'obj':
                self.read_obj(self.filename)
            else:
                if self.warning:
                    print('Not support file type')

    def get_mesh_data_from_vtkPolyData(self):
        data = self.vtkPolyData

        n_triangles = data.GetNumberOfCells()
        n_points = data.GetNumberOfPoints()
        mesh_triangles = np.zeros([n_triangles, 9], dtype='float32')
        mesh_triangle_ids = np.zeros([n_triangles, 3], dtype='int32')
        mesh_points = np.zeros([n_points, 3], dtype='float32')

        for i in range(n_triangles):
            mesh_triangles[i][0], mesh_triangles[i][1], mesh_triangles[i][2] = data.GetPoint(
                data.GetCell(i).GetPointId(0))
            mesh_triangles[i][3], mesh_triangles[i][4], mesh_triangles[i][5] = data.GetPoint(
                data.GetCell(i).GetPointId(1))
            mesh_triangles[i][6], mesh_triangles[i][7], mesh_triangles[i][8] = data.GetPoint(
                data.GetCell(i).GetPointId(2))
            mesh_triangle_ids[i][0] = data.GetCell(i).GetPointId(0)
            mesh_triangle_ids[i][1] = data.GetCell(i).GetPointId(1)
            mesh_triangle_ids[i][2] = data.GetCell(i).GetPointId(2)

        for i in range(n_points):
            mesh_points[i][0], mesh_points[i][1], mesh_points[i][2] = data.GetPoint(i)

        self.cells = mesh_triangles
        self.cell_ids = mesh_triangle_ids
        self.points = mesh_points

        # read point arrays
        for i_attribute in range(self.vtkPolyData.GetPointData().GetNumberOfArrays()):
            #            print(self.vtkPolyData.GetPointData().GetArrayName(i_attribute))
            #            print(self.vtkPolyData.GetPointData().GetArray(i_attribute).GetNumberOfComponents())
            self.load_point_attributes(self.vtkPolyData.GetPointData().GetArrayName(i_attribute),
                                       self.vtkPolyData.GetPointData().GetArray(i_attribute).GetNumberOfComponents())

        # read cell arrays
        for i_attribute in range(self.vtkPolyData.GetCellData().GetNumberOfArrays()):
            #            print(self.vtkPolyData.GetCellData().GetArrayName(i_attribute))
            #            print(self.vtkPolyData.GetCellData().GetArray(i_attribute).GetNumberOfComponents())
            self.load_cell_attributes(self.vtkPolyData.GetCellData().GetArrayName(i_attribute),
                                      self.vtkPolyData.GetCellData().GetArray(i_attribute).GetNumberOfComponents())

    def read_stl(self, stl_filename):
        '''
        update
            self.filename
            self.reader
            self.vtkPolyData
            self.cells
            self.cell_ids
            self.points
            self.cell_attributes
            self.point_attributes
        '''
        #        self.filename = stl_filename
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_filename)
        reader.Update()
        self.reader = reader

        data = reader.GetOutput()
        self.vtkPolyData = data
        self.get_mesh_data_from_vtkPolyData()

    def read_obj(self, obj_filename):
        '''
        update
            self.filename
            self.reader
            self.vtkPolyData
            self.cells
            self.cell_ids
            self.points
            self.cell_attributes
            self.point_attributes
        '''
        #        self.filename = obj_filename
        reader = vtk.vtkOBJReader()
        reader.SetFileName(obj_filename)
        reader.Update()
        self.reader = reader

        data = reader.GetOutput()
        self.vtkPolyData = data
        self.get_mesh_data_from_vtkPolyData()

    def read_vtp(self, vtp_filename):
        '''
        update
            self.filename
            self.reader
            self.vtkPolyData
            self.cells
            self.cell_ids
            self.points
            self.cell_attributes
            self.point_attributes
        '''
        #        self.filename = vtp_filename
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(vtp_filename)
        reader.Update()
        self.reader = reader

        data = reader.GetOutput()
        self.vtkPolyData = data
        self.get_mesh_data_from_vtkPolyData()

    def load_point_attributes(self, attribute_name, dim):
        self.point_attributes[attribute_name] = np.zeros([self.points.shape[0], dim], dtype='float32')
        try:
            if dim == 1:
                for i in range(self.points.shape[0]):
                    self.point_attributes[attribute_name][i, 0] = self.vtkPolyData.GetPointData().GetArray(
                        attribute_name).GetValue(i)
            elif dim == 2:
                for i in range(self.points.shape[0]):
                    self.point_attributes[attribute_name][i, 0] = self.vtkPolyData.GetPointData().GetArray(
                        attribute_name).GetComponent(i, 0)
                    self.point_attributes[attribute_name][i, 1] = self.vtkPolyData.GetPointData().GetArray(
                        attribute_name).GetComponent(i, 1)
            elif dim == 3:
                for i in range(self.points.shape[0]):
                    self.point_attributes[attribute_name][i, 0] = self.vtkPolyData.GetPointData().GetArray(
                        attribute_name).GetComponent(i, 0)
                    self.point_attributes[attribute_name][i, 1] = self.vtkPolyData.GetPointData().GetArray(
                        attribute_name).GetComponent(i, 1)
                    self.point_attributes[attribute_name][i, 2] = self.vtkPolyData.GetPointData().GetArray(
                        attribute_name).GetComponent(i, 2)
        except:
            if self.warning:
                print('No cell attribute named "{0}" in file: {1}'.format(attribute_name, self.filename))

    def get_point_curvatures(self, method='mean'):
        curv = vtk.vtkCurvatures()
        curv.SetInputData(self.vtkPolyData)
        if method == 'mean':
            curv.SetCurvatureTypeToMean()
        elif method == 'max':
            curv.SetCurvatureTypeToMaximum()
        elif method == 'min':
            curv.SetCurvatureTypeToMinimum()
        elif method == 'Gaussian':
            curv.SetCurvatureTypeToGaussian()
        else:
            curv.SetCurvatureTypeToMean()
        curv.Update()

        n_points = self.vtkPolyData.GetNumberOfPoints()
        self.point_attributes['Curvature'] = np.zeros([n_points, 1], dtype='float32')
        for i in range(n_points):
            self.point_attributes['Curvature'][i] = curv.GetOutput().GetPointData().GetArray(0).GetValue(i)

    def get_cell_curvatures(self, method='mean'):
        self.get_point_curvatures(method=method)
        self.cell_attributes['Curvature'] = np.zeros([self.cells.shape[0], 1], dtype='float32')

        # optimized way
        tmp_cell_curvts = self.point_attributes['Curvature'][self.cell_ids].squeeze()
        self.cell_attributes['Curvature'] = np.mean(tmp_cell_curvts, axis=-1).reshape([tmp_cell_curvts.shape[0], 1])

    def load_cell_attributes(self, attribute_name, dim):
        self.cell_attributes[attribute_name] = np.zeros([self.cells.shape[0], dim], dtype='float32')
        try:
            if dim == 1:
                for i in range(self.cells.shape[0]):
                    self.cell_attributes[attribute_name][i, 0] = self.vtkPolyData.GetCellData().GetArray(
                        attribute_name).GetValue(i)
            elif dim == 2:
                for i in range(self.cells.shape[0]):
                    self.cell_attributes[attribute_name][i, 0] = self.vtkPolyData.GetCellData().GetArray(
                        attribute_name).GetComponent(i, 0)
                    self.cell_attributes[attribute_name][i, 1] = self.vtkPolyData.GetCellData().GetArray(
                        attribute_name).GetComponent(i, 1)
            elif dim == 3:
                for i in range(self.cells.shape[0]):
                    self.cell_attributes[attribute_name][i, 0] = self.vtkPolyData.GetCellData().GetArray(
                        attribute_name).GetComponent(i, 0)
                    self.cell_attributes[attribute_name][i, 1] = self.vtkPolyData.GetCellData().GetArray(
                        attribute_name).GetComponent(i, 1)
                    self.cell_attributes[attribute_name][i, 2] = self.vtkPolyData.GetCellData().GetArray(
                        attribute_name).GetComponent(i, 2)
        except:
            if self.warning:
                print('No cell attribute named "{0}" in file: {1}'.format(attribute_name, self.filename))

    def set_cell_labels(self, label_dict, tol=0.01):
        '''
        update:
            self.cell_attributes['Label']
        '''
        self.cell_attributes['Label'] = np.zeros([self.cell_ids.shape[0], 1], dtype='float32')

        cell_centers = (self.cells[:, 0:3] + self.cells[:, 3:6] + self.cells[:, 6:9]) / 3.0
        for i_label in label_dict:
            i_label_cell_centers = (label_dict[i_label][:, 0:3] + label_dict[i_label][:, 3:6] + label_dict[i_label][:,
                                                                                                6:9]) / 3.0
            D = distance_matrix(cell_centers, i_label_cell_centers)
            if len(np.argwhere(D <= tol)) > i_label_cell_centers.shape[0]:
                sys.exit('tolerance ({0}) is too large, please adjust.'.format(tol))
            elif len(np.argwhere(D <= tol)) < i_label_cell_centers.shape[0]:
                sys.exit('tolerance ({0}) is too small, please adjust.'.format(tol))
            else:
                for i in range(i_label_cell_centers.shape[0]):
                    label_id = np.argwhere(D <= tol)[i][0]
                    self.cell_attributes['Label'][label_id, 0] = int(i_label)

    def set_cell_labels_map(self, label_dict, tol=0.5, refine=False, missing=False):  # 0.01
        '''
        label dict: dict of {31, cells of that tooth (n by 9 array)}, {32, cells of that tooth}, etc.
        update:
            self.cell_attributes['Label']
        '''
        # consider partial mapping
        from scipy.spatial import distance_matrix
        # consider partial mapping
        labelledInput = True
        if not 'Label' in self.cell_attributes.keys():
            labelledInput = False
            self.cell_attributes['Label'] = np.zeros([self.cell_ids.shape[0], 1], dtype='float32')

        original_labelling = self.cell_attributes['Label'].copy()
        self.cell_attributes['Distance'] = np.ones([self.cell_ids.shape[0], 1], dtype='float32')
        cell_centers = (self.cells[:, 0:3] + self.cells[:, 3:6] + self.cells[:, 6:9]) / 3.0
        for i_label in label_dict:
            i_label_cell_centers = (label_dict[i_label][:, 0:3] + label_dict[i_label][:, 3:6] + label_dict[i_label][:,
                                                                                                6:9]) / 3.0
            D = distance_matrix(cell_centers, i_label_cell_centers)
            Shortest_D = np.min(D, axis=1)
            # compute the shortest distance
            shortest_index = sorted(range(len(Shortest_D)), key=lambda k: Shortest_D[k])
            K_shortest_index = shortest_index[0:math.ceil(i_label_cell_centers.shape[0])]
            # print(Shortest_D[K_shortest_index])
            Shortest_D.sort()
            # print(Shortest_D)
            # Update the label if you find the better label with the shortest path

            for i, label_id in enumerate(K_shortest_index):
                if self.cell_attributes['Distance'][label_id, 0] >= Shortest_D[i] and (Shortest_D[i] < tol):
                    # self.cell_attributes['Label'][label_id, 0] = int(i_label)+1
                    self.cell_attributes['Label'][label_id, 0] = int(i_label)
                    self.cell_attributes['Distance'][label_id, 0] = Shortest_D[i]

        if refine:
            if labelledInput:
                idx = []
                for i in range(original_labelling.shape[0]):
                    if original_labelling[i, 0] != self.cell_attributes['Label'][i, 0]:
                        idx.append(i)
                diff_labelling = original_labelling[idx]
                print('diff_labelling shape = ', diff_labelling.shape)
            else:
                diff_labelling = np.empty(shape=(0, 0))
            # Generate the patch_prob_output        
            labels = np.unique(self.cell_attributes['Label'][:, 0])
            num_classes = len(labels)

            # need to be relabelled before GC refine
            possible_labels = sorted(np.unique(self.cell_attributes[attribute_name][:0]))
            labels_dict = {}

            for i in range(len(possible_labels)):
                labels_dict[i] = possible_labels[i]
                idx_tooth = np.where(self.cell_attributes[attribute_name] == possible_labels[i])[0]
                self.cell_attributes[attribute_name][idx_tooth] = i

            labels_d = self.cell_attributes['Label']
            labels_d = [int(i[0]) for i in labels_d]

            # Generate the patch_prob_output
            Num_cells = self.cells.shape[0]
            patch_prob_output = np.zeros([1, Num_cells, num_classes], dtype='float32')
            patch_prob_output[0, :][list(range(0, Num_cells)), labels_d] = 1

            self.graph_cut_refinement(patch_prob_output, diff_labelling)
            self.relabelling_with_dict(labels_dict)

    def Depth_Img(self):
        cell_centers = (self.cells[:, 0:3] + self.cells[:, 3:6] + self.cells[:, 6:9]) / 3.0
        max_z = np.int32(max(cell_centers[:, 2]))
        depth_z = max_z - cell_centers[:, 2]
        cell_centers = np.int32(cell_centers)
        IMG_z = np.zeros([100, 100])
        depth_z = (255 * (depth_z - np.min(depth_z)) / np.ptp(depth_z)).astype(int)
        IMG_z[cell_centers[:, 0], cell_centers[:, 1]] = depth_z
        # Normalised [0,255] as integer: don't forget the parenthesis before astype(int)
        im_z = Image.fromarray(IMG_z)
        plt.imshow(im_z, cmap=plt.get_cmap('gray'))
        plt.savefig('lena_greyscale.png')
        plt.show()

    def get_distance_for_sub_division(self, ref_labels):
        """
        compute the distance from one side of a tooth to the same side of the farthest tooth
        @param ref_labels:
        @return:
        """
        assert len(ref_labels) > 1, "More than 1 labels is needed to compute the distance"
        first_label = min(ref_labels)
        second_label = max(ref_labels)

        # get the teeth
        first_tooth_idx = [idx for idx, e in enumerate(self.cell_attributes['Label']) if e in [first_label]]
        second_tooth_idx = [idx for idx, e in enumerate(self.cell_attributes['Label']) if e in [second_label]]
        first_tooth_cell_centers = (self.cells[first_tooth_idx, 0:3] + self.cells[first_tooth_idx, 3:6] + self.cells[first_tooth_idx,
                                                                                        6:9]) / 3.0
        second_tooth_cell_centers = (self.cells[second_tooth_idx, 0:3] + self.cells[second_tooth_idx, 3:6] + self.cells[second_tooth_idx,
                                                                                        6:9]) / 3.0

        # compute the shortest distance
        D = distance_matrix(first_tooth_cell_centers, second_tooth_cell_centers)
        max_D = np.max(D, axis=1)
        return np.min(max_D)

    def Sub_divided_mesh(self, ref_labels, idx, Mesh_name, tol, tol_for_all_teeth=False, output_path=''):
        """
        Save a part of a mesh with the ref labels + gingiva
        @param ref_labels: list of integers, the teeth you want ex: (5, 6, 7)
        @param idx: int, part number in the saving string
        @param Mesh_name: str, name for the saving
        @param tol: float, distance threshold to select the gingiva
        @param tol_for_all_teeth: bool, apply the tol on the min distance between a point and all the teeth (True) or the farthest tooth (False)
        @return: None, a file is save in Output_path
        """
        Part_mesh = Easy_Mesh()

        # create a jaw mesh
        mesh_jaw = Easy_Mesh()
        jaw_idx = np.where(self.cell_attributes['Label'] == 0)[0]  # extract the target label
        mesh_jaw.cells = self.cells[jaw_idx]
        mesh_jaw.update_cell_ids_and_points()
        jaw_cell_center = (self.cells[jaw_idx, 0:3] + self.cells[jaw_idx, 3:6] + self.cells[jaw_idx, 6:9]) / 3.0

        teeth_idx = [idx for idx, e in enumerate(self.cell_attributes['Label']) if e in ref_labels]

        if tol_for_all_teeth:
            # get the teeth (together)
            teeth_cell_centers = (self.cells[teeth_idx, 0:3] + self.cells[teeth_idx, 3:6] + self.cells[teeth_idx, 6:9]) / 3.0

            # compute the shortest distance
            D = distance_matrix(jaw_cell_center, teeth_cell_centers)
            min_D = np.min(D, axis=1)
            sorted_min_index = sorted(range(len(min_D)), key=lambda k: min_D[k])
            min_D.sort()
            idx_max = np.max(np.where(min_D < tol), axis=1)

        else:
            # the threshold need to be satisfied for all the teeth separately
            every_min_D = []
            for label in ref_labels:
                # get the tooth
                tooth_idx = [idx for idx, e in enumerate(self.cell_attributes['Label']) if e in [label]]
                tooth_cell_centers = (self.cells[tooth_idx, 0:3] + self.cells[tooth_idx, 3:6] + self.cells[tooth_idx, 6:9]) / 3.0

                # compute the shortest distance
                D = distance_matrix(jaw_cell_center, tooth_cell_centers)
                min_D = np.min(D, axis=1)
                every_min_D.append(min_D)

            # Use the min distance from every jaw cell to it's farthest tooth
            min_D = np.max(np.vstack(every_min_D), axis=0)
            sorted_min_index = sorted(range(len(min_D)), key=lambda k: min_D[k])
            min_D.sort()
            idx_max = np.max(np.where(min_D < tol), axis=1)

        # save the part of mesh
        Part_mesh.cells = mesh_jaw.cells[sorted_min_index[0:idx_max[0]]]
        Part_mesh.cells = np.append(Part_mesh.cells, self.cells[teeth_idx], axis=0)
        Part_mesh.update_cell_ids_and_points()
        Part_mesh.cell_attributes['Label'] = np.zeros([len(teeth_idx) + idx_max[0], 1], dtype=np.int32)
        Part_mesh.cell_attributes['Label'][idx_max[0]:] = self.cell_attributes['Label'][teeth_idx]
        save_string = os.path.join(output_path, 'Part_{}_size_{}_{}'.format(idx, len(ref_labels), Mesh_name))
        Part_mesh.to_vtp(save_string)
        return save_string

    def close_tooth_side_by_replacement(self, tooth):
        tooth_label = tooth.cell_attributes['Label'][0][0]

        # remove original tooth and add the exocad one
        new_mesh = Easy_Mesh()
        other_label_idx = np.where(self.cell_attributes['Label'] != tooth_label)[0]
        new_mesh.cells = np.append(self.cells[other_label_idx], tooth.cells, axis=0)
        new_mesh.update_cell_ids_and_points()
        new_mesh.cell_attributes['Label'] = np.concatenate((self.cell_attributes['Label'][other_label_idx], tooth.cell_attributes['Label']), axis=0)
        new_mesh.to_vtp('test_mesh_tooth_{}_replaced.vtp'.format(tooth_label))

        # update easy mesh
        self.cells = new_mesh.cells
        self.update_cell_ids_and_points()
        self.cell_attributes['Label'] = new_mesh.cell_attributes['Label']
        return None

    def assign_cell_labels(self, label):
        '''
        update:
            self.cell_attributes['Label']
        '''
        self.cell_attributes['Label'] = int(label) * np.ones([self.cell_ids.shape[0], 1])

    def get_cell_edges(self):
        '''
        update:
            self.cell_attributes['Edge']
        '''
        self.cell_attributes['Edge'] = np.zeros([self.cell_ids.shape[0], 3], dtype='float32')

        for i_count in range(self.cell_ids.shape[0]):
            v1 = self.points[self.cell_ids[i_count, 0], :] - self.points[self.cell_ids[i_count, 1], :]
            v2 = self.points[self.cell_ids[i_count, 1], :] - self.points[self.cell_ids[i_count, 2], :]
            v3 = self.points[self.cell_ids[i_count, 0], :] - self.points[self.cell_ids[i_count, 2], :]
            self.cell_attributes['Edge'][i_count, 0] = np.linalg.norm(v1)
            self.cell_attributes['Edge'][i_count, 1] = np.linalg.norm(v2)
            self.cell_attributes['Edge'][i_count, 2] = np.linalg.norm(v3)

    def get_cell_normals(self):
        data = self.vtkPolyData
        n_triangles = data.GetNumberOfCells()
        # normal
        v1 = np.zeros([n_triangles, 3], dtype='float32')
        v2 = np.zeros([n_triangles, 3], dtype='float32')
        v1[:, 0] = self.cells[:, 0] - self.cells[:, 3]
        v1[:, 1] = self.cells[:, 1] - self.cells[:, 4]
        v1[:, 2] = self.cells[:, 2] - self.cells[:, 5]
        v2[:, 0] = self.cells[:, 3] - self.cells[:, 6]
        v2[:, 1] = self.cells[:, 4] - self.cells[:, 7]
        v2[:, 2] = self.cells[:, 5] - self.cells[:, 8]
        mesh_normals = np.cross(v1, v2)
        mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
        mesh_normals[:, 0] /= mesh_normal_length[:]
        mesh_normals[:, 1] /= mesh_normal_length[:]
        mesh_normals[:, 2] /= mesh_normal_length[:]
        self.cell_attributes['Normal'] = mesh_normals

    def compute_guassian_heatmap(self, landmark, sigma=10.0, height=1.0):
        '''
        inputs:
            landmark: np.array [1, 3]
            sigma (default=10.0)
            height (default=1.0)
        update:
            self.cell_attributes['heatmap']
        '''
        cell_centers = (self.cells[:, 0:3] + self.cells[:, 3:6] + self.cells[:, 6:9]) / 3.0
        heatmap = np.zeros([cell_centers.shape[0], 1], dtype='float32')

        for i_cell in range(len(cell_centers)):
            delx = cell_centers[i_cell, 0] - landmark[0]
            dely = cell_centers[i_cell, 1] - landmark[1]
            delz = cell_centers[i_cell, 2] - landmark[2]
            heatmap[i_cell, 0] = height * math.exp(-1 * (delx * delx + dely * dely + delz * delz) / 2.0 / sigma / sigma)
        self.cell_attributes['Heatmap'] = heatmap

    def compute_displacement_map(self, landmark):
        '''
        inputs:
            landmark: np.array [1, 3]
        update:
            self.cell_attributes['Displacement map']
        '''
        cell_centers = (self.cells[:, 0:3] + self.cells[:, 3:6] + self.cells[:, 6:9]) / 3.0
        displacement_map = np.zeros([cell_centers.shape[0], 3], dtype='float32')

        for i_cell in range(len(cell_centers)):
            delx = cell_centers[i_cell, 0] - landmark[0]
            dely = cell_centers[i_cell, 1] - landmark[1]
            delz = cell_centers[i_cell, 2] - landmark[2]
            displacement_map[i_cell, 0] = delx
            displacement_map[i_cell, 1] = dely
            displacement_map[i_cell, 2] = delz
        self.cell_attributes['Displacement_map'] = displacement_map

    def get_labelled_points_from_cells(self):
        try:
            labels = self.cell_attributes['Label']
        except:
            labels = None
        middle_points = (self.cells[:, :3] + self.cells[:, 3:6] + self.cells[:, 6:]) / 3
        return labels, middle_points

    def relabelling_with_dict(self, labels_dict, compact=False):
        '''
        inputs: 
            labels_dict: labelling dictionary used to update existing labelling obtained with graph_cut_refinement
        update: self.cell_attributes['Label']
        '''
        possible_labels = np.unique(self.cell_attributes['Label'][:, 0])
        labels = sorted(labels_dict.keys(), reverse=not compact)
        for i in labels:
            if i in possible_labels:
                # print('assigne label ', i, 'as ', labels_dict[i])
                self.cell_attributes['Label'][self.cell_attributes['Label'] == i] = labels_dict[i]

    def compute_cell_attributes_by_svm(self, given_cells, given_cell_attributes, attribute_name, refine=False):
        '''
        inputs:
            given_cells: [n, 9] numpy array
            given_cell_attributes: [n, 1] numpy array
        update:
            self.cell_attributes[attribute_name]
        '''
        if given_cell_attributes.shape[1] == 1:
            # create a dict for final labelling mapping in case of discontinous labelling
            self.cell_attributes[attribute_name] = np.zeros([self.cells.shape[0], 1], dtype='float32')
            if refine:
                clf = svm.SVC(probability=True)
            else:
                clf = svm.SVC()
            clf.fit(given_cells, given_cell_attributes.ravel())
            self.cell_attributes[attribute_name][:, 0] = clf.predict(self.cells)

            if refine:
                # need to be relabelled before GC refine
                possible_labels = sorted(np.unique(self.cell_attributes[attribute_name][:0]))
                labels_dict = {}

                for i in range(len(possible_labels)):
                    labels_dict[i] = possible_labels[i]
                    idx_tooth = np.where(self.cell_attributes[attribute_name] == possible_labels[i])[0]
                    self.cell_attributes[attribute_name][idx_tooth] = i
                self.cell_attributes[attribute_name + '_proba'] = clf.predict_proba(self.cells)
                self.graph_cut_refinement(self.cell_attributes[attribute_name + '_proba'])
                self.relabelling_with_dict(labels_dict)
        else:
            if self.warning:
                print('Only support 1D attribute')

    def compute_cell_attributes_by_knn(self, given_cells, given_cell_attributes, attribute_name, k=3, refine=False):
        '''
        inputs:
            given_cells: [n, 9] numpy array
            given_cell_attributes: [n, 1] numpy array
        update:
            self.cell_attributes[attribute_name]
        '''
        if given_cell_attributes.shape[1] == 1:
            self.cell_attributes[attribute_name] = np.zeros([self.cells.shape[0], 1], dtype='float32')
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(given_cells, given_cell_attributes.ravel())
            self.cell_attributes[attribute_name][:, 0] = neigh.predict(self.cells)
            self.cell_attributes[attribute_name + '_proba'] = neigh.predict_proba(self.cells)
            if refine:
                # need to be relabelled before GC refine
                possible_labels = sorted(np.unique(self.cell_attributes[attribute_name][:, 0]))
                print('after knn possible labels = ', possible_labels)
                labels_dict = {}

                for i in range(len(possible_labels)):
                    labels_dict[i] = possible_labels[i]
                    idx_tooth = np.where(self.cell_attributes[attribute_name] == possible_labels[i])[0]
                    # print('label = ', i , ' nb cells = ', len(idx_tooth))
                    self.cell_attributes[attribute_name][idx_tooth] = i
                # print('labels_dict = ', labels_dict)
                self.graph_cut_refinement(self.cell_attributes[attribute_name + '_proba'])
                self.relabelling_with_dict(labels_dict)

        else:
            if self.warning:
                print('Only support 1D attribute')

    def graph_cut_refinement(self, patch_prob_output, modifiable=np.empty(shape=(0, 0))):
        round_factor = 100
        patch_prob_output[patch_prob_output < 1.0e-6] = 1.0e-6
        labels = np.unique(self.cell_attributes['Label'][:, 0])
        num_classes = patch_prob_output.shape[-1]
        # unaries
        unaries = -round_factor * np.log10(patch_prob_output)
        unaries = unaries.astype(np.int32)
        unaries = unaries.reshape(-1, num_classes)
        # parawise
        pairwise = (1 - np.eye(num_classes, dtype=np.int32))
        # edges
        self.get_cell_normals()
        normals = self.cell_attributes['Normal'][:]
        cells = self.cells[:]
        cell_ids = self.cell_ids[:]
        barycenters = (cells[:, 0:3] + cells[:, 3:6] + cells[:, 6:9]) / 3.0

        lambda_c = 30
        edges = np.empty([1, 3], order='C')
        for i_node in range(cells.shape[0]):
            # Find neighbors
            nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
            nei_id = np.where(nei == 2)
            for i_nei in nei_id[0][:]:
                if i_node < i_nei:
                    cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]) / np.linalg.norm(
                        normals[i_node, 0:3]) / np.linalg.norm(normals[i_nei, 0:3])
                    if cos_theta >= 1.0:
                        cos_theta = 0.9999
                    theta = np.arccos(cos_theta)
                    phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
                    if theta > np.pi / 2.0:
                        edges = np.concatenate(
                            (edges, np.array([i_node, i_nei, -math.log10(theta / np.pi) * phi]).reshape(1, 3)), axis=0)
                    else:
                        beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                        edges = np.concatenate(
                            (edges, np.array([i_node, i_nei, -beta * math.log10(theta / np.pi) * phi]).reshape(1, 3)),
                            axis=0)
        edges = np.delete(edges, 0, 0)
        edges[:, 2] *= lambda_c * round_factor
        edges = edges.astype(np.int32)
        refine_labels = cut_from_graph(edges, unaries, pairwise)
        refine_labels = refine_labels.reshape([-1, 1])

        # we need a mapping here to take account of discontinous labelling
        # refine_labels suppose a continous labelling
        # output refined result
        for i in range(self.cells.shape[0]):
            if modifiable.shape[0] > 0:
                if i in modifiable[:, 0]:
                    self.cell_attributes['Label'][i] = refine_labels[i]
            else:
                self.cell_attributes['Label'] = refine_labels

    def update_cell_ids_and_points(self):
        '''
        call when self.cells is modified
        update
            self.cell_ids
            self.points
        '''
        rdt_points = self.cells.reshape([int(self.cells.shape[0] * 3), 3])
        self.points, idx = np.unique(rdt_points, return_inverse=True, axis=0)
        self.cell_ids = idx.reshape([-1, 3])

        if self.warning:
            print('Warning! self.cell_attributes are reset and need to be updated!')
        self.cell_attributes = dict()  # reset
        self.point_attributes = dict()  # reset
        self.update_vtkPolyData()

    def update_vtkPolyData(self):
        '''
        call this function when manipulating self.cells, self.cell_ids, or self.points
        '''
        vtkPolyData = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()

        points.SetData(numpy_to_vtk(self.points))
        cells.SetCells(len(self.cell_ids),
                       numpy_to_vtkIdTypeArray(np.hstack((np.ones(len(self.cell_ids))[:, None] * 3,
                                                          self.cell_ids)).astype(np.int64).ravel(),
                                               deep=1))
        vtkPolyData.SetPoints(points)
        vtkPolyData.SetPolys(cells)

        # update point_attributes
        for i_key in self.point_attributes.keys():
            point_attribute = vtk.vtkDoubleArray()
            point_attribute.SetName(i_key);
            if self.point_attributes[i_key].shape[1] == 1:
                point_attribute.SetNumberOfComponents(self.point_attributes[i_key].shape[1])
                for i_attribute in self.point_attributes[i_key]:
                    point_attribute.InsertNextTuple1(i_attribute)
                vtkPolyData.GetPointData().AddArray(point_attribute)
            #                vtkPolyData.GetPointData().SetScalars(cell_attribute)
            elif self.point_attributes[i_key].shape[1] == 2:
                point_attribute.SetNumberOfComponents(self.point_attributes[i_key].shape[1])
                for i_attribute in self.point_attributes[i_key]:
                    point_attribute.InsertNextTuple2(i_attribute[0], i_attribute[1])
                vtkPolyData.GetPointData().AddArray(point_attribute)
            #                vtkPolyData.GetPointData().SetVectors(cell_attribute)
            elif self.point_attributes[i_key].shape[1] == 3:
                point_attribute.SetNumberOfComponents(self.point_attributes[i_key].shape[1])
                for i_attribute in self.point_attributes[i_key]:
                    point_attribute.InsertNextTuple3(i_attribute[0], i_attribute[1], i_attribute[2])
                vtkPolyData.GetPointData().AddArray(point_attribute)
            #                vtkPolyData.GetPointData().SetVectors(cell_attribute)
            else:
                if self.warning:
                    print('Check attribute dimension, only support 1D, 2D, and 3D now')

        # update cell_attributes
        for i_key in self.cell_attributes.keys():
            cell_attribute = vtk.vtkDoubleArray()
            cell_attribute.SetName(i_key);
            if self.cell_attributes[i_key].shape[1] == 1:
                cell_attribute.SetNumberOfComponents(self.cell_attributes[i_key].shape[1])
                for i_attribute in self.cell_attributes[i_key]:
                    cell_attribute.InsertNextTuple1(i_attribute)
                vtkPolyData.GetCellData().AddArray(cell_attribute)
            #                vtkPolyData.GetCellData().SetScalars(cell_attribute)
            elif self.cell_attributes[i_key].shape[1] == 2:
                cell_attribute.SetNumberOfComponents(self.cell_attributes[i_key].shape[1])
                for i_attribute in self.cell_attributes[i_key]:
                    cell_attribute.InsertNextTuple2(i_attribute[0], i_attribute[1])
                vtkPolyData.GetCellData().AddArray(cell_attribute)
            #                vtkPolyData.GetCellData().SetVectors(cell_attribute)
            elif self.cell_attributes[i_key].shape[1] == 3:
                cell_attribute.SetNumberOfComponents(self.cell_attributes[i_key].shape[1])
                for i_attribute in self.cell_attributes[i_key]:
                    cell_attribute.InsertNextTuple3(i_attribute[0], i_attribute[1], i_attribute[2])
                vtkPolyData.GetCellData().AddArray(cell_attribute)
            #                vtkPolyData.GetCellData().SetVectors(cell_attribute)
            else:
                if self.warning:
                    print('Check attribute dimension, only support 1D, 2D, and 3D now')

        vtkPolyData.Modified()
        self.vtkPolyData = vtkPolyData

    def extract_largest_region(self):
        connect = vtk.vtkPolyDataConnectivityFilter()
        connect.SetInputData(self.vtkPolyData)
        connect.SetExtractionModeToLargestRegion()
        connect.Update()

        self.vtkPolyData = connect.GetOutput()
        self.get_mesh_data_from_vtkPolyData()
        if self.warning:
            print('Warning! self.cell_attributes are reset and need to be updated!')
        self.cell_attributes = dict()  # reset
        self.point_attributes = dict()  # reset

    def merge_vertices(self):
        print('before merge_vertices points size = ', self.points.shape[0])
        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputData(self.vtkPolyData)
        cleanFilter.ConvertStripsToPolysOff()
        cleanFilter.ConvertPolysToLinesOff()
        cleanFilter.ConvertLinesToPointsOff()
        cleanFilter.PointMergingOn()
        cleanFilter.Update()
        self.vtkPolyData = cleanFilter.GetOutput()
        self.get_mesh_data_from_vtkPolyData()
        print('after merge_vertices points size = ', self.points.shape[0])
        if self.warning:
            print('Warning! self.cell_attributes are reset and need to be updated!')
        self.cell_attributes = dict()  # reset
        self.point_attributes = dict()  # reset

    def mesh_decimation(self, reduction_rate, original_label_status=False, refine=False):
        # check mesh has label attribute or not
        if original_label_status:
            original_cells = self.cells.copy()
            original_labels = self.cell_attributes['Label'].copy()
        self.merge_vertices()
        decimate_reader = vtk.vtkQuadricDecimation()
        decimate_reader.SetInputData(self.vtkPolyData)
        decimate_reader.SetTargetReduction(reduction_rate)
        decimate_reader.VolumePreservationOn()
        decimate_reader.AttributeErrorMetricOn()
        decimate_reader.NormalsAttributeOn()
        decimate_reader.Update()
        self.vtkPolyData = decimate_reader.GetOutput()
        self.get_mesh_data_from_vtkPolyData()
        if self.warning:
            print('Warning! self.cell_attributes are reset and need to be updated!')
        self.cell_attributes = dict()  # reset
        self.point_attributes = dict()  # reset

        # if original_label_status:
        #    self.compute_cell_attributes_by_svm(original_cells, original_labels, 'Label')
        if original_label_status:
            self.compute_cell_attributes_by_knn(original_cells, original_labels, 'Label', 3, refine)

    def mesh_subdivision(self, num_subdivisions, method='loop', original_label_status=False):
        if method == 'loop':
            subdivision_reader = vtk.vtkLoopSubdivisionFilter()
        elif method == 'butterfly':
            subdivision_reader = vtk.vtkButterflySubdivisionFilter()
        else:
            if self.warning:
                print('Not a valid subdivision method')

        # check mesh has label attribute or not
        if original_label_status:
            original_cells = self.cells.copy()
            original_labels = self.cell_attributes['Label'].copy()

        subdivision_reader.SetInputData(self.vtkPolyData)
        subdivision_reader.SetNumberOfSubdivisions(num_subdivisions)
        subdivision_reader.Update()
        self.vtkPolyData = subdivision_reader.GetOutput()
        self.get_mesh_data_from_vtkPolyData()
        if self.warning:
            print('Warning! self.cell_attributes are reset and need to be updated!')
        self.cell_attributes = dict()  # reset
        self.point_attributes = dict()  # reset

        if original_label_status:
            self.compute_cell_attributes_by_svm(original_cells, original_labels, 'Label')

    def mesh_transform(self, vtk_matrix):
        Trans = vtk.vtkTransform()
        Trans.SetMatrix(vtk_matrix)

        TransFilter = vtk.vtkTransformPolyDataFilter()
        TransFilter.SetTransform(Trans)
        TransFilter.SetInputData(self.vtkPolyData)
        TransFilter.Update()

        self.vtkPolyData = TransFilter.GetOutput()
        self.get_mesh_data_from_vtkPolyData()

    def mesh_reflection(self, ref_axis='x', STDLabelling=False):
        '''
        This function is only for tooth arch model,
        it will flip the label (n=15 so far) as well.
        input:
            ref_axis: 'x'/'y'/'z'
        '''

        RefFilter = vtk.vtkReflectionFilter()
        if ref_axis == 'x':
            RefFilter.SetPlaneToX()
        elif ref_axis == 'y':
            RefFilter.SetPlaneToY()
        elif ref_axis == 'z':
            RefFilter.SetPlaneToZ()
        else:
            if self.warning:
                print('Invalid ref_axis!')

        RefFilter.CopyInputOff()
        RefFilter.SetInputData(self.vtkPolyData)
        RefFilter.Update()

        self.vtkPolyData = RefFilter.GetOutput()
        self.get_mesh_data_from_vtkPolyData()
        # need to be reviewed for STD labelling
        original_cell_labels = np.copy(self.cell_attributes['Label'])  # add original cell label back
        # validate mesh labelling
        labels = np.unique(self.cell_attributes['Label'][:, 0])
        if np.min(labels[1:-1]) > 10 and not STDLabelling:
            print('Warning: input mesh could be in STD labelling, check it!')
        elif np.min(labels[1:-1]) < 10 and STDLabelling:
            print('Warning: input mesh is not in STD labelling, check it!')
        else:
            if STDLabelling:
                # Upperjaw
                for i in range(11, 19):
                    if len(original_cell_labels == i) > 0:
                        self.cell_attributes['Label'][original_cell_labels == i] = i + 10
                for i in range(21, 29):
                    if len(original_cell_labels == i) > 0:
                        self.cell_attributes['Label'][original_cell_labels == i] = i - 10
                # Lowerjaw
                for i in range(31, 39):
                    if len(original_cell_labels == i) > 0:
                        self.cell_attributes['Label'][original_cell_labels == i] = i + 10
                for i in range(41, 49):
                    if len(original_cell_labels == i) > 0:
                        self.cell_attributes['Label'][original_cell_labels == i] = i - 10
                # Upper prep
                for i in range(111, 119):
                    if len(original_cell_labels == i) > 0:
                        self.cell_attributes['Label'][original_cell_labels == i] = i + 10
                for i in range(121, 129):
                    if len(original_cell_labels == i) > 0:
                        self.cell_attributes['Label'][original_cell_labels == i] = i - 10
                # Lower prep
                for i in range(131, 139):
                    if len(original_cell_labels == i) > 0:
                        self.cell_attributes['Label'][original_cell_labels == i] = i + 10
                for i in range(141, 149):
                    if len(original_cell_labels == i) > 0:
                        self.cell_attributes['Label'][original_cell_labels == i] = i - 10
            else:  # for meshSegNet labelling
                # for permanent teeth: we consider only 16 teeth for each arch.
                for i in range(1, 15):
                    if len(original_cell_labels == i) > 0:
                        self.cell_attributes['Label'][
                            original_cell_labels == i] = 15 - i  # 1 -> 14, 2 -> 13, ..., 14 -> 1
                # for wisdom teeth: exchange the label 15 and 16
                if len(original_cell_labels == 15) > 0:
                    self.cell_attributes['Label'][original_cell_labels == 15] = 16
                if len(original_cell_labels == 16) > 0:
                    self.cell_attributes['Label'][original_cell_labels == 16] = 15

    def get_boundary_points(self):
        '''
        output: boundary_points [n, 3] nparray
        '''
        featureEdges = vtk.vtkFeatureEdges()
        featureEdges.SetInputData(self.vtkPolyData)
        featureEdges.BoundaryEdgesOn()
        featureEdges.FeatureEdgesOff()
        featureEdges.ManifoldEdgesOff()
        featureEdges.NonManifoldEdgesOff()
        featureEdges.Update()

        num_bps = featureEdges.GetOutput().GetNumberOfPoints()
        boundary_points = np.zeros([num_bps, 3], dtype='float32')
        for i in range(num_bps):
            boundary_points[i][0], boundary_points[i][1], boundary_points[i][2] = featureEdges.GetOutput().GetPoint(i)

        return boundary_points

    def to_vtp(self, vtp_filename):
        self.update_vtkPolyData()

        if vtk.VTK_MAJOR_VERSION <= 5:
            self.vtkPolyData.Update()

        writer = vtk.vtkXMLPolyDataWriter();
        writer.SetFileName("{0}".format(vtp_filename));

        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(self.vtkPolyData)
        else:
            writer.SetInputData(self.vtkPolyData)

        writer.Write()

    def to_stl(self, stl_filename):
        self.update_vtkPolyData()

        if vtk.VTK_MAJOR_VERSION <= 5:
            self.vtkPolyData.Update()

        writer = vtk.vtkSTLWriter()
        writer.SetFileName(stl_filename)
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(self.vtkPolyData)
        else:
            writer.SetInputData(self.vtkPolyData)
        writer.Write()

    def to_obj(self, obj_filename):
        self.merge_vertices()
        self.update_vtkPolyData()
        with open(obj_filename, 'w') as f:
            for i_point in self.points:
                f.write("v {} {} {}\n".format(i_point[0], i_point[1], i_point[2]))

            if not 'Lable' in self.cell_attributes.keys():
                self.assign_cell_labels(0)
            for i_label in np.unique(self.cell_attributes['Label']):
                f.write("g mmGroup{}\n".format(int(i_label)))
                label_cell_ids = np.where(self.cell_attributes['Label'] == i_label)[0]
                for i_label_cell_id in label_cell_ids:
                    i_cell = self.cell_ids[i_label_cell_id]
                    f.write(
                        "f {}//{} {}//{} {}//{}\n".format(i_cell[0] + 1, i_cell[0] + 1, i_cell[1] + 1, i_cell[1] + 1,
                                                          i_cell[2] + 1, i_cell[2] + 1))


# ------------------------------------------------------------------------------
def GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                               translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                               scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]):
    '''
    get transformation matrix (4*4)

    return: vtkMatrix4x4
    '''

    Trans = vtk.vtkTransform()
    # seed = 4321
    # np.random.seed(seed)
    ry_flag = np.random.randint(0, 2)  # if 0, no rotate
    rx_flag = np.random.randint(0, 2)  # if 0, no rotate
    rz_flag = np.random.randint(0, 2)  # if 0, no rotate
    if ry_flag == 1:
        # rotate along Yth axis
        Trans.RotateY(np.random.uniform(rotate_Y[0], rotate_Y[1]))
    if rx_flag == 1:
        # rotate along Xth axis
        Trans.RotateX(np.random.uniform(rotate_X[0], rotate_X[1]))
    if rz_flag == 1:
        # rotate along Zth axis
        Trans.RotateZ(np.random.uniform(rotate_Z[0], rotate_Z[1]))

    trans_flag = np.random.randint(0, 2)  # if 0, no translate
    if trans_flag == 1:
        Trans.Translate([np.random.uniform(translate_X[0], translate_X[1]),
                         np.random.uniform(translate_Y[0], translate_Y[1]),
                         np.random.uniform(translate_Z[0], translate_Z[1])])

    scale_flag = np.random.randint(0, 2)
    if scale_flag == 1:
        Trans.Scale([np.random.uniform(scale_X[0], scale_X[1]),
                     np.random.uniform(scale_Y[0], scale_Y[1]),
                     np.random.uniform(scale_Z[0], scale_Z[1])])

    matrix = Trans.GetMatrix()

    return matrix


def listComplementElements(list1, list2):
    storeResults = []
    for num in list1:
        if num not in list2:  # this will essentially iterate your list behind the scenes
            storeResults.append(num)
    return storeResults


def subtract_two_meshes(Closed_mesh, Open_mesh, tol=0.01):  # 0.01
    '''
    update:
        self.cell_attributes['Label']
    '''
    cell_centers_open = (Open_mesh.cells[:, 0:3] + Open_mesh.cells[:, 3:6] + Open_mesh.cells[:, 6:9]) / 3.0
    cell_centers_cloesd = (Closed_mesh.cells[:, 0:3] + Closed_mesh.cells[:, 3:6] + Closed_mesh.cells[:, 6:9]) / 3.0
    D = distance_matrix(cell_centers_cloesd, cell_centers_open)
    Shortest_D = np.min(D, axis=1)
    # compute the shortest distance
    K_shortest_index = sorted(range(len(Shortest_D)), key=lambda k: Shortest_D[k])
    # print(Shortest_D[K_shortest_index])
    Shortest_D.sort()
    # print(Shortest_D)
    # Update the label if you find the better label with the shortest path
    List_index = []
    for i, label_id in enumerate(K_shortest_index):
        if (Shortest_D[i] < tol):
            List_index.append(K_shortest_index[i])
    Closed_mesh.cells = np.delete(Closed_mesh.cells, List_index, 0)
    Closed_mesh.update_cell_ids_and_points()
    return Closed_mesh


def Mapping_two_meshes(source_mesh, target_mesh, tol=0.01):  # 0.01
    '''
    return cloest cells (distance < tol) of target_mesh from source_mesh
    '''
    cell_centers_source = (source_mesh.cells[:, 0:3] + source_mesh.cells[:, 3:6] + source_mesh.cells[:, 6:9]) / 3.0
    cell_centers_target = (target_mesh.cells[:, 0:3] + target_mesh.cells[:, 3:6] + target_mesh.cells[:, 6:9]) / 3.0
    D = distance_matrix(cell_centers_target, cell_centers_source)
    Shortest_D = np.min(D, axis=1)
    # compute the shortest distance
    K_shortest_index = sorted(range(len(Shortest_D)), key=lambda k: Shortest_D[k])
    # print(Shortest_D[K_shortest_index])
    Shortest_D.sort()
    # print(Shortest_D)
    # Update the label if you find the better label with the shortest path
    List_index = []
    for i, label_id in enumerate(K_shortest_index):
        if (Shortest_D[i] < tol):
            List_index.append(K_shortest_index[i])
    if len(List_index) == 0:
        print('Warning: tol is too small for mesh mapping, shortedt distance = ', Shortest_D[0])
    mesh_tmp = Easy_Mesh()
    mesh_tmp.cells = target_mesh.cells[List_index]
    mesh_tmp.update_cell_ids_and_points()
    return mesh_tmp


def RemoveTooth(original_mesh, missing_label):
    '''
    remove from original_mesh a tooth labelled as label_id
    return: new mesh without tooth labelled as label_id
    '''
    Tooth_path = './'
    repaired_path = './'

    mesh_tmp = Easy_Mesh()

    # Extract the tooth
    i_tooth_idx = np.where(original_mesh.cell_attributes['Label'] == missing_label)[0]  # extract the target label
    i_mesh = Easy_Mesh()
    i_mesh.cells = original_mesh.cells[i_tooth_idx]
    i_mesh.update_cell_ids_and_points()
    i_mesh.cell_attributes['Label'] = np.zeros([len(i_tooth_idx), 1], dtype=np.int32)  # create cell array
    i_mesh.cell_attributes['Label'][
    :] = missing_label  # assign the correct label to the cell_array, although it's useless in an obj file
    i_mesh.to_stl(os.path.join(Tooth_path, 'tooth_open_sides{}.stl'.format(missing_label)))
    # Fill tooth's side
    Open_mesh = pv.read(os.path.join(Tooth_path, 'tooth_open_sides{}.stl'.format(missing_label)))

    # fill crown
    mfix = PyTMesh(False)
    mfix.load_file(os.path.join(Tooth_path, 'tooth_open_sides{}.stl'.format(missing_label)))
    mfix.fill_small_boundaries(nbe=1000, refine=True)

    vert, faces = mfix.return_arrays()
    triangles = np.empty((faces.shape[0], 4))
    triangles[:, -3:] = faces
    triangles[:, 0] = 3

    mesh = pv.PolyData(vert, triangles.astype(int))
    mesh.save(os.path.join(Tooth_path, 'tooth_closed_sides{}.stl'.format(missing_label)))

    Open_mesh = Easy_Mesh(os.path.join(Tooth_path, 'tooth_open_sides{}.stl'.format(missing_label)))
    Closed_mesh = Easy_Mesh(os.path.join(Tooth_path, 'tooth_closed_sides{}.stl'.format(missing_label)))
    # add the new tooth to original mesh
    # Keep the sides ..remove the extras
    Closed_sides = subtract_two_meshes(Closed_mesh, Open_mesh)
    Closed_sides.to_stl(os.path.join(Tooth_path, 'tooth_sides_before_inverting_normals{}.stl'.format(missing_label)))
    # invert the normal of Closed_sides
    normals = pv.PolyData(os.path.join(Tooth_path, 'tooth_sides_before_inverting_normals{}.stl'.format(missing_label)))
    normals.flip_normals()
    normals.save(os.path.join(Tooth_path, 'tooth_sides_after_inverting_normals{}.stl'.format(missing_label)))
    # Add sides to removed tooth from original mesh
    mesh_tmp.cells = np.delete(original_mesh.cells, i_tooth_idx, 0)
    mesh_tmp.update_cell_ids_and_points()
    # mesh_tmp.to_stl(os.path.join(repaired_path, 'mesh_after_extracted_{}.stl'.format(missing_label)))
    Closed_sides = Easy_Mesh(
        os.path.join(Tooth_path, 'tooth_sides_after_inverting_normals{}.stl'.format(missing_label)))
    mesh_tmp.cells = np.concatenate((mesh_tmp.cells, Closed_sides.cells), axis=0)
    mesh_tmp.update_cell_ids_and_points()
    mesh_tmp.to_stl(os.path.join(repaired_path, 'mesh_filled_after_inverting_normals_{}.stl'.format(missing_label)))
    # Refine the filled mesh
    mfix = PyTMesh(False)
    mfix.load_file(os.path.join(repaired_path, 'mesh_filled_after_inverting_normals_{}.stl'.format(missing_label)))
    # fill holes
    mfix.fill_small_boundaries(nbe=100, refine=True)

    vert, faces = mfix.return_arrays()

    triangles = np.empty((faces.shape[0], 4))
    triangles[:, -3:] = faces
    triangles[:, 0] = 3

    # remove all intermediate files
    os.remove(os.path.join(Tooth_path, 'tooth_open_sides{}.stl'.format(missing_label)))
    os.remove(os.path.join(Tooth_path, 'tooth_closed_sides{}.stl'.format(missing_label)))
    os.remove(os.path.join(Tooth_path, 'tooth_sides_before_inverting_normals{}.stl'.format(missing_label)))
    os.remove(os.path.join(Tooth_path, 'tooth_sides_after_inverting_normals{}.stl'.format(missing_label)))
    os.remove(os.path.join(repaired_path, 'mesh_filled_after_inverting_normals_{}.stl'.format(missing_label)))

    mesh = pv.PolyData(vert, triangles.astype(int))
    mesh.save(os.path.join(repaired_path, 'Mesh_with_missing{}.stl'.format(missing_label)))


def get_mesh_with_removed_tooth(original_mesh, missing_label):
    RemoveTooth(original_mesh, missing_label)
    part_of_mesh_without_tooth = Easy_Mesh('Mesh_with_missing{}.stl'.format(missing_label))
    os.remove('Mesh_with_missing{}.stl'.format(missing_label))
    return part_of_mesh_without_tooth


def get_tooth(mesh, label):
    """
    @param mesh: easy mesh object
    @param label: int, msn label of the tooth to extract
    @return: easy_mesh object with only the triangles of the tooth
    """
    tooth_idx = np.where(mesh.cell_attributes['Label'] == label)[0]  # extract the target label
    tooth = Easy_Mesh()
    tooth.cells = mesh.cells[tooth_idx]
    tooth.update_cell_ids_and_points()
    tooth.cell_attributes['Label'] = np.zeros([len(tooth_idx), 1], dtype=np.int32)
    tooth.cell_attributes['Label'][:] = label
    # tooth.to_stl('tooth_open_sides{}.stl'.format(label))
    return tooth


def RemoveTeeth(original_mesh, missing_labels):
    """
    remove from original_mesh a tooth labelled as missing_labels
    original_mesh : Easy_mesh object
    missing_labels : list of int
    """
    assert type(missing_labels) == list, 'missing_labels argument should be of type list, use RemoveTooth for int'

    Tooth_path = ''
    repaired_path = ''

    for missing_label in missing_labels:
        # Create the tooth
        tooth_idx = np.where(original_mesh.cell_attributes['Label'] == missing_label)[0]  # extract the target label
        tooth = Easy_Mesh()
        tooth.cells = original_mesh.cells[tooth_idx]
        tooth.update_cell_ids_and_points()
        tooth.cell_attributes['Label'] = np.ones([len(tooth_idx), 1], dtype=np.int32) * missing_label
        tooth.to_stl(os.path.join(Tooth_path, 'tooth_open_sides{}.stl'.format(missing_label)))

        # Close the sides
        mfix = PyTMesh(False)
        mfix.load_file(os.path.join(Tooth_path, 'tooth_open_sides{}.stl'.format(missing_label)))
        mfix.fill_small_boundaries(nbe=2000, refine=True)

        vert, faces = mfix.return_arrays()
        triangles = np.empty((faces.shape[0], 4))
        triangles[:, -3:] = faces
        triangles[:, 0] = 3

        mesh = pv.PolyData(vert, triangles.astype(int))
        mesh.save(os.path.join(Tooth_path, 'tooth_closed_sides{}.stl'.format(missing_label)))

        # Get the sides only
        Open_mesh = Easy_Mesh(os.path.join(Tooth_path, 'tooth_open_sides{}.stl'.format(missing_label)))
        Closed_mesh = Easy_Mesh(os.path.join(Tooth_path, 'tooth_closed_sides{}.stl'.format(missing_label)))
        Closed_sides = subtract_two_meshes(Closed_mesh, Open_mesh)
        Closed_sides.to_stl(
            os.path.join(Tooth_path, 'tooth_sides_before_inverting_normals{}.stl'.format(missing_label)))

        # invert the normals
        normals = pv.PolyData(
            os.path.join(Tooth_path, 'tooth_sides_before_inverting_normals{}.stl'.format(missing_label)))
        normals.flip_normals()
        normals.save(os.path.join(Tooth_path, 'tooth_sides_after_inverting_normals{}.stl'.format(missing_label)))

        # remove all intermediate files
        os.remove(os.path.join(Tooth_path, 'tooth_open_sides{}.stl'.format(missing_label)))
        os.remove(os.path.join(Tooth_path, 'tooth_closed_sides{}.stl'.format(missing_label)))
        os.remove(os.path.join(Tooth_path, 'tooth_sides_before_inverting_normals{}.stl'.format(missing_label)))

    # Remove tooth from original mesh
    tooth_idx = []
    for missing_label in missing_labels:
        tooth_idx = np.concatenate((tooth_idx, np.where(original_mesh.cell_attributes['Label'] == missing_label)[0]),
                                   axis=0)  # extract the target label
    tooth_idx = np.sort(tooth_idx)
    original_mesh.cells = np.delete(original_mesh.cells, tooth_idx.astype(int), 0)
    original_mesh.update_cell_ids_and_points()

    # Fill mesh with inverted sides
    for missing_label in missing_labels:
        Closed_sides = Easy_Mesh(
            os.path.join(Tooth_path, 'tooth_sides_after_inverting_normals{}.stl'.format(missing_label)))
        original_mesh.cells = np.concatenate((original_mesh.cells, Closed_sides.cells), axis=0)

    original_mesh.update_cell_ids_and_points()
    original_mesh.to_stl(
        os.path.join(repaired_path, 'mesh_filled_after_inverting_normals_{}.stl'.format(str(missing_labels))))

    # fill holes
    mfix = PyTMesh(False)
    mfix.load_file(
        os.path.join(repaired_path, 'mesh_filled_after_inverting_normals_{}.stl'.format(str(missing_labels))))
    mfix.fill_small_boundaries(nbe=100, refine=True)

    vert, faces = mfix.return_arrays()
    triangles = np.empty((faces.shape[0], 4))
    triangles[:, -3:] = faces
    triangles[:, 0] = 3

    mesh = pv.PolyData(vert, triangles.astype(int))
    output_file = 'Mesh_with_missing_'
    for i in missing_labels:
        output_file += str(i) + '_'
    output_file = output_file[0:-1] + '.stl'
    mesh.save(os.path.join(repaired_path, output_file))
    os.remove(os.path.join(repaired_path, 'mesh_filled_after_inverting_normals_{}.stl'.format(missing_labels)))
    return None


def AppendMeshes(dict_label_files, output_filename, reduction_rate=0):
    '''
    append a list of meshes in stl, obj or vtp, with a reduce_rate for decimation
    output a combined mesh in vtk format
    '''
    append = vtk.vtkAppendPolyData()
    for i in dict_label_files:
        if isinstance(dict_label_files[i], tuple):
            (filename, reduction_rate) = dict_label_files[i]
        else:
            filename = dict_label_files[i]

        mesh = Easy_Mesh(filename)
        if (reduction_rate > 0 and reduction_rate < 1.0):
            decimate_reader = vtk.vtkQuadricDecimation()
            decimate_reader.SetInputData(mesh.vtkPolyData)
            decimate_reader.SetTargetReduction(reduction_rate)
            decimate_reader.VolumePreservationOn()
            decimate_reader.Update()
            mesh.vtkPolyData = decimate_reader.GetOutput()
            mesh.get_mesh_data_from_vtkPolyData()
        else:
            print("No decimation for the mesh with Label %d" % i)
        mesh.assign_cell_labels(int(i))
        # this line could be reviewed. We delete GroupIds attribute just for meshlaber to deal with Label correctly  
        # mesh.cell_attributes.pop('GroupIds')
        mesh.update_vtkPolyData()
        if vtk.VTK_MAJOR_VERSION <= 5:
            mesh.vtkPolyData.Update()
        append.AddInputData(mesh.vtkPolyData)
    append.Update()

    # remove any duplicate points
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(append.GetOutput())
    clean.Update()

    # output final combined mesh with labels
    writer = vtk.vtkXMLPolyDataWriter();
    writer.SetFileName(output_filename)

    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(append.GetOutput())
    else:
        writer.SetInputData(append.GetOutput())
    writer.Write()


def AppendMeshesWithTargetSize(dict_label_files, output_filename, target_size, percentage_gingiva):
    '''
    append a list of meshes in stl, obj or vtp 
    output a combined mesh in vtk format with an approximate size of target_size
    '''

    # we suppose the gingiva is always with label 0
    mesh_sizes = []
    for i in dict_label_files:
        filename = dict_label_files[i]
        mesh = Easy_Mesh(filename)
        mesh_sizes.append(mesh.cells.shape[0])

    gingiva_size = mesh_sizes[0]
    assert (percentage_gingiva > 0 and percentage_gingiva < 1.0), "Invalid percentage_gingiva!"

    gingiva_target_size = percentage_gingiva * target_size
    gingiva_decimate_rate = 1.0 - gingiva_target_size / float(gingiva_size) - 0.005
    print('Gingiva decimate_rate will be %f for a target mesh size of %d and original mesh size of %d' % (
        gingiva_decimate_rate, gingiva_target_size, gingiva_size))
    rest_mesh_size = sum(mesh_sizes) - gingiva_size
    rest_target_size = (1.0 - percentage_gingiva) * target_size
    # Ammar's suggestion
    reduction_rate = 1.0 - rest_target_size / float(rest_mesh_size) - 0.005
    assert (reduction_rate > 0 and percentage_gingiva < 1.0), "Invalid decimation rate obtained!"
    print('Decimate_rate will be %f for a target mesh size of %d and original mesh size of %d' % (
        reduction_rate, rest_target_size, rest_mesh_size))
    new_dict = {}
    for i in dict_label_files:
        if i == 0:
            new_dict[i] = (dict_label_files[i], gingiva_decimate_rate)
        else:
            new_dict[i] = (dict_label_files[i], reduction_rate)

    AppendMeshes(new_dict, output_filename)


def RemoveToothInFile(input_filename, label, output_filename1, output_filename2):
    '''
    remove tooth labelled with 'label' from the mesh given by input_filename
    output a mesh without tooth labelled as 'label' 
    '''
    mesh = Easy_Mesh(input_filename)

    if 'Label' in mesh.cell_attributes.keys():
        ids = vtk.vtkIdTypeArray()
        ids.SetNumberOfComponents(1)
        # Specify that we want to extract cells 
        for cell_id in range(mesh.cell_ids.shape[0]):
            if mesh.cell_attributes['Label'][cell_id, 0] == int(label):
                ids.InsertNextValue(cell_id)

        selectionNode = vtk.vtkSelectionNode()
        selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
        selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
        selectionNode.SetSelectionList(ids)

        selection = vtk.vtkSelection()
        selection.AddNode(selectionNode)

        extractSelection = vtk.vtkExtractSelection()

        if vtk.VTK_MAJOR_VERSION <= 5:
            extractSelection.SetInput(0, mesh.vtkPolyData)
            extractSelection.SetInput(1, selection)
        else:
            extractSelection.SetInputData(0, mesh.vtkPolyData)
            extractSelection.SetInputData(1, selection)
        extractSelection.Update()

        extract_surface = vtk.vtkDataSetSurfaceFilter()
        extract_surface.SetInputData(extractSelection.GetOutput())
        extract_surface.Update()

        writer_selected = vtk.vtkXMLPolyDataWriter();
        writer_selected.SetFileName(output_filename1)
        writer_selected.SetInputData(extract_surface.GetOutput())
        writer_selected.Write()

        # Get points that are NOT in the selection
        # invert the selection
        selectionNode.GetProperties().Set(vtk.vtkSelectionNode.INVERSE(), 1)
        extractSelection.Update()

        extract_surface = vtk.vtkDataSetSurfaceFilter()
        extract_surface.SetInputData(extractSelection.GetOutput())
        extract_surface.Update()

        writer_notSelected = vtk.vtkXMLPolyDataWriter();
        writer_notSelected.SetFileName(output_filename2)
        writer_notSelected.SetInputData(extract_surface.GetOutput())
        writer_notSelected.Write()

    else:
        print("Input mesh without 'Label' cell attribute!")


def getMeshSegNetLabelling():
    '''
    convert from std labelling to meshSegNet labelling
    '''

    alias = {
        17: 1, 37: 1, 117: 1, 137: 1,
        16: 2, 36: 2, 116: 2, 136: 2,
        15: 3, 35: 3, 115: 3, 135: 3,
        14: 4, 34: 4, 114: 4, 134: 4,
        13: 5, 33: 5, 113: 5, 133: 5,
        12: 6, 32: 6, 112: 6, 132: 6,
        11: 7, 31: 7, 111: 7, 131: 7,
        21: 8, 41: 8, 121: 8, 141: 8,
        22: 9, 42: 9, 122: 9, 142: 9,
        23: 10, 43: 10, 123: 10, 143: 10,
        24: 11, 44: 11, 124: 11, 144: 11,
        25: 12, 45: 12, 125: 12, 145: 12,
        26: 13, 46: 13, 126: 13, 146: 13,
        27: 14, 47: 14, 127: 14, 147: 14,
        28: 15, 48: 15, 128: 15, 148: 15,
        18: 16, 38: 16, 118: 16, 138: 16
    }

    return alias


def generateSTDLabellingDict(archType, listPrep):
    MappingMeshSegNetLabellingToSTDlabelling = {
        1: 17,
        2: 16,
        3: 15,
        4: 14,
        5: 13,
        6: 12,
        7: 11,
        8: 21,
        9: 22,
        10: 23,
        11: 24,
        12: 25,
        13: 26,
        14: 27,
        15: 28,
        16: 18
    }
    # labelling prep with prefix '1'
    if len(listPrep) > 0:
        for label in listPrep:
            MappingMeshSegNetLabellingToSTDlabelling[label] += 100
    if archType == 'Lower':
        for label in MappingMeshSegNetLabellingToSTDlabelling.keys():
            MappingMeshSegNetLabellingToSTDlabelling[label] += 20
    return MappingMeshSegNetLabellingToSTDlabelling


def fromMeshSegNetLabellingToSTDLabelling(mesh, archType, listPrep=[]):
    '''
    update: mesh.cell_attributes['Label']
    '''
    labels = np.unique(mesh.cell_attributes['Label'][:, 0])
    print('Possible labels: ', labels)
    labelling_dict = generateSTDLabellingDict(archType, listPrep)
    mesh.relabelling_with_dict(labelling_dict)
    return mesh


def fromMeshSegNetLabellingToSTDLabellingWithSTDReferenceMesh(mesh, meshSTD):
    '''
    update: mesh.cell_attributes['Label']
    '''
    labels = np.unique(mesh.cell_attributes['Label'][:, 0])
    # print('Possible labels: ', labels)

    # get archType and listPrep from meshSTD
    labels_std = np.unique(meshSTD.cell_attributes['Label'][:, 0])
    archType = 'Upper'

    for label in labels_std:
        if label > 0 and label < 100:
            if label > 30:
                archType = 'Lower'
                break
    listPrep = []
    dict = getMeshSegNetLabelling()
    for label in labels_std:
        if label > 100:
            listPrep.append(dict[label])

    labelling_dict = generateSTDLabellingDict(archType, listPrep)
    mesh.relabelling_with_dict(labelling_dict)
    return mesh


def fromSTDLabellingToMeshSegNetLabelling(mesh):
    '''
    update: mesh.cell_attributes['Label']
    '''
    labels = np.unique(mesh.cell_attributes['Label'][:, 0])
    # print('Possible labels: ', labels)
    labelling_dict = getMeshSegNetLabelling()
    mesh.relabelling_with_dict(labelling_dict, compact=True)
    return mesh


def getLabellingDictForMeshSegNet(TeethLabels):
    '''
    return a labelling dict for MeshSegNet
    '''
    meshSegNetLabellingDict = {
        18: 16,
        17: 1,
        16: 2,
        15: 3,
        14: 4,
        13: 5,
        12: 6,
        11: 7,
        21: 8,
        22: 9,
        23: 10,
        24: 11,
        25: 12,
        26: 13,
        27: 14,
        28: 15,
        38: 16,
        37: 1,
        36: 2,
        35: 3,
        34: 4,
        33: 5,
        32: 6,
        31: 7,
        41: 8,
        42: 9,
        43: 10,
        44: 11,
        45: 12,
        46: 13,
        47: 14,
        48: 15
    }

    label_dict = {}
    label_dict[0] = 'gingiva.obj'
    for label in TeethLabels:
        label_dict[meshSegNetLabellingDict[label[0:2]]] = label + '.obj'
    return label_dict


def get_msn_to_std_labeling_dict():
    msn_to_std_labelling = {
        1: 17,
        2: 16,
        3: 15,
        4: 14,
        5: 13,
        6: 12,
        7: 11,
        8: 21,
        9: 22,
        10: 23,
        11: 24,
        12: 25,
        13: 26,
        14: 27,
        15: 28,
        16: 18
    }
    return msn_to_std_labelling
