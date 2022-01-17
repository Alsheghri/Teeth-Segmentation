# This code will augment the data in small ranges because they are already registered

from easy_mesh_vtk import *

if __name__ == "__main__":

    data_path = 'C:/Users/Poly/OneDrive - polymtl.ca/Documents - DentalAI/Données/Data to publish/Experiment/Code/SelfSupervised/Supervised Train registered_vtp'

    # Augmentation
    mesh_list = [i for i in os.listdir(data_path) if
                  os.path.isfile(os.path.join(data_path, i)) and 'vtp' in i]
    num_augmentations = 20

    for file_name in mesh_list:
        for i_aug in range(num_augmentations):
            output_file_name = file_name.split('.')[0] + '_A{0}'.format(i_aug) + '.vtp'
            vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-10, 10], rotate_Y=[-10, 10], rotate_Z=[-10, 10],
                                                    translate_X=[-1, 1], translate_Y=[-1, 1], translate_Z=[-1, 1],
                                                    scale_X=[0.9, 1.1], scale_Y=[0.9, 1.1],
                                                    scale_Z=[0.9, 1.1])  # use default random setting

            mesh = Easy_Mesh(os.path.join(data_path, file_name))
            mesh.mesh_transform(vtk_matrix)
            mesh.to_vtp(os.path.join(data_path, output_file_name))
