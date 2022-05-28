from enum import IntEnum
import pyvista as pv
import numpy as np


class VT(IntEnum):
     Air = 0
     Wood = 1
     Leaves = 2



def voxel(O, voxel_type, p, a=0.8):
    """
    :param O: Точка основания куба
    :param voxel_type: тип куба
    :param a: длина ребра
    :param p:plotter
    """

    center = O + a/2
    cube = pv.Cube(center=center, x_length=a, y_length=a, z_length=a)
    if voxel_type == VT.Air:
        pass
        # p.add_mesh(cube, style='wireframe', color='blue')
    elif voxel_type == VT.Leaves:
        p.add_mesh(cube, color='green')
    else:
        p.add_mesh(cube, color='brown')


# max_Z, max_Y, max_X = 14, 30, 29


def read_cloud():
    with open('viz_creation.npy', 'rb') as f:
        viz = np.load(f, allow_pickle=True)
    return viz


def main():
    p = pv.Plotter()
    viz = read_cloud()
    print("Load completed")
    z = 0
    for viz_z in viz:
        print(f"Progress Z: {z}")
        y = 0
        for viz_y in viz_z:
            x = 0
            for viz_x in viz_y:
                voxel(np.array([z, y, x]), viz_x, p)
                x += 1
            y += 1
        z += 1

    p.show()


if __name__ == '__main__':
    main()
