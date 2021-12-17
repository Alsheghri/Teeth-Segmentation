# This code includes the transformation method.
# We are transforming the upper arches to have the same orientation as lower.
# The end result should have the z_range to minimum. The Z normal to be positive and maximum and
# the Y normal to be positive. But the x_normal could be positive or negative.

import os
from easy_mesh_vtk import *


def transformZ(archName, rotateAngle):
    mesh = Easy_Mesh(os.path.join(output_save_path, archName))
    Trans = vtk.vtkTransform()
    Trans.RotateZ(rotateAngle)
    vtk_matrix = Trans.GetMatrix()
    mesh.mesh_transform(vtk_matrix)
    output_file_name = i.split('.')[0] + '.vtp'
    mesh.to_vtp(os.path.join(output_save_path, output_file_name))
    #output_file_name = i.split('.')[0] + '_rotatedZ.obj'
    #mesh.to_obj(os.path.join(output_save_path, output_file_name))

def transformY(archName, rotateAngle) -> object:
    mesh = Easy_Mesh(os.path.join(output_save_path, archName))
    Trans = vtk.vtkTransform()
    Trans.RotateY(rotateAngle)
    vtk_matrix = Trans.GetMatrix()
    mesh.mesh_transform(vtk_matrix)
    output_file_name = i.split('.')[0] + '.vtp'
    mesh.to_vtp(os.path.join(output_save_path, output_file_name))
    #output_file_name = i.split('.')[0] + '_rotatedY.obj'
    #mesh.to_obj(os.path.join(output_save_path, output_file_name))

def transformX(archName, rotateAngle):
    mesh = Easy_Mesh(os.path.join(output_save_path, archName))
    Trans = vtk.vtkTransform()
    Trans.RotateX(rotateAngle)
    vtk_matrix = Trans.GetMatrix()
    mesh.mesh_transform(vtk_matrix)
    output_file_name = i.split('.')[0] + '.vtp'
    mesh.to_vtp(os.path.join(output_save_path, output_file_name))
    #output_file_name = i.split('.')[0] + '_rotatedX.obj'
    #mesh.to_obj(os.path.join(output_save_path, output_file_name))

def transformXZ(archName, xRotatreAngle, zRotateAngle):
    mesh = Easy_Mesh(os.path.join(output_save_path, archName))
    Trans = vtk.vtkTransform()
    Trans.RotateX(xRotatreAngle)
    Trans.RotateZ(zRotateAngle)
    vtk_matrix = Trans.GetMatrix()
    mesh.mesh_transform(vtk_matrix)
    output_file_name = i.split('.')[0] + '.vtp'
    mesh.to_vtp(os.path.join(output_save_path, output_file_name))
    #output_file_name = i.split('.')[0] + '_rotatedXZ.obj'
    #mesh.to_obj(os.path.join(output_save_path, output_file_name))

def transformZY(archName, zRotatreAngle, yRotateAngle):
    mesh = Easy_Mesh(os.path.join(output_save_path, archName))
    Trans = vtk.vtkTransform()
    Trans.RotateZ(zRotatreAngle)
    Trans.RotateY(yRotateAngle)
    vtk_matrix = Trans.GetMatrix()
    mesh.mesh_transform(vtk_matrix)
    output_file_name = i.split('.')[0] + '.vtp'
    mesh.to_vtp(os.path.join(output_save_path, output_file_name))
    #output_file_name = i.split('.')[0] + '_rotatedZY.obj'
    #mesh.to_obj(os.path.join(output_save_path, output_file_name))


if __name__ == "__main__":

    # Reflection

    #output_save_path = './augmentation_vtk_data/'
    output_save_path = './Transformation/'


    if not os.path.exists(output_save_path):
        os.mkdir(output_save_path)

    vtk_path = output_save_path
    mesh_list_upper = [i for i in os.listdir(output_save_path) if
                 os.path.isfile(os.path.join(output_save_path, i)) and 'vtp' in i and 'Upper' in i]
    mesh_list_lower = [i for i in os.listdir(output_save_path) if
                 os.path.isfile(os.path.join(output_save_path, i)) and 'vtp' in i and 'Lower' in i]
    mesh_list = [i for i in os.listdir(output_save_path) if
                 os.path.isfile(os.path.join(output_save_path, i)) and 'vtp' in i]

    # Rotation to bring upper to the same orientation of lower

    for i in mesh_list:
        print(i)
        mesh = Easy_Mesh(os.path.join(output_save_path, i))
        normals = mesh.get_cell_normals()
        x_coordinates = [*mesh.cells[:, 0], *mesh.cells[:, 3], *mesh.cells[:, 6]]
        y_coordinates = [*mesh.cells[:, 1], *mesh.cells[:, 4], *mesh.cells[:, 7]]
        z_coordinates = [*mesh.cells[:, 2], *mesh.cells[:, 5], *mesh.cells[:, 8]]
        x_range = max(x_coordinates) - min(x_coordinates)
        y_range = max(y_coordinates) - min(y_coordinates)
        z_range = max(z_coordinates) - min(z_coordinates)
        normals = mesh.cell_attributes['Normal']
        x_normals = sum(normals[:, 0])
        y_normals = sum(normals[:, 1])
        z_normals = sum(normals[:, 2])
        # TODO need to make the transformation below more universal 
        if 'Upper' in i:
            if z_range == min([x_range, z_range, y_range]):
                if z_normals < 0 and y_normals > 0 and y_range < x_range:
                    transformY(i, 180)
                if z_normals < 0 and y_normals < 0:
                    transformZY(i, 180, 180)
                if z_normals > 0 and y_normals < 0:
                    transformZ(i, 180)
                if z_normals < 0 and x_normals < 0 and y_normals > 0 and y_range > x_range:
                    transformXZ(i, 180, 90)

            else:
            # z_range != min([x_range, z_range, y_range]):
                if y_normals < 0 and y_normals == min([x_normals, y_normals, z_normals]):
                    transformX(i, 270)
                if y_normals > z_normals:
                    transformXZ(i, 270, 180)

        if 'Lower' in i:
            if z_range != min([x_range, z_range, y_range]):
                if y_normals == max([x_normals, y_normals, z_normals]):
                    transformXZ(i, 270, 180)