"""
Deformably register two point clouds, extracted from vtk files, then write deformably registered point cloud to
new vtk file using original triangulation information. Also includes functions to transform ablation site information
to model coordinate system using initial and transformed Ensite point clouds. 

Authors: Julie Shade, Pranak Lakshiminarayanan
Last edited: 6/23/17

"""


def read_point_cloud_from_vtk_file(filename):
    """
    Reads a vtk file
    :param filename: 
    :return: 
    """
    pass

def register_pc(moving, fixed):
    """
    Deformable registers poi
    :param pc1: 
    :param pc2: 
    :return: 
    """
    pass

def transform_ablation_sites():
    """
    Transforms ablation locations (given in point cloud form) from Ensite coordinate system to model
    coordinate system using thin plate splines extracted from initial and transformed point clouds representations of 
    Ensite surface.
    
    :return: 
    """
    pass

def pc_to_vtk(pc, infile, outfile):
    """
    Converts a point cloud to .vtk format using triangulation information from an input vtk file, writes result 
    to output .vtk file.
    
    :param pc: 
    :param infile: 
    :param outfile: 
    :return: 
    """
    pass

def main():
   print("this is working ye")

if __name__ == '__main__':
    main()