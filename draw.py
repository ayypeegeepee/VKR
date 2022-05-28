import os
import open3d as o3d
import numpy as np
from enum import IntEnum


class VT(IntEnum):
    Air = 0
    Wood = 1
    Leaves = 2


def read_cloud(filepath):
    with open(filepath, 'rb') as f:
        viz = np.load(f, allow_pickle=True)
    return viz


def voxel(O, voxel_type):
    d = 0.8
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=d,
                                                    height=d,
                                                    depth=d)
    # mesh_box.compute_vertex_normals()
    mesh_box.translate(O)
    if voxel_type == VT.Air:
        pass
        # p.add_mesh(cube, style='wireframe', color='blue')
    elif voxel_type == VT.Leaves:
        mesh_box.paint_uniform_color([0.1, 0.9, 0.1])
    else:
        mesh_box.paint_uniform_color([0.1, 0.1, 0.7])

    return mesh_box


def main():
    # viz = read_cloud("tmp\\viz_creation.npy")
    # viz = read_cloud("viz_creation.npy")
    # viz = read_cloud("tmp\\voxelized.npy")
    viz = read_cloud("data\\tree01\\voxelized.npy")

    print("Load completed")
    meshes = []

    z = 0
    for viz_z in viz:
        print(f"Progress Z: {z}")
        y = 0
        for viz_y in viz_z:
            x = 0
            for viz_x in viz_y:
                if viz_x > 0:
                    meshes.append(voxel(np.array([z, y, x]), viz_x))
                x += 1
            y += 1
        z += 1
    o3d.visualization.draw_geometries(meshes)


if __name__ == "__main__":
    main()
