from enum import IntEnum
import numpy as np
import open3d as o3d
from pathlib import Path

from cubeforest import CubeForestModel, VoxelMap
from cubeforest.categories import Wood

PC_FILE_EXTENSION = ".ply"


class VT(IntEnum):
    Air = 0
    Wood = 1
    Leaves = 2


def p(points_file):
    input_cloud = o3d.io.read_point_cloud(str(points_file))
    return np.asarray(input_cloud.points)


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
    full_points_file = Path("data") / "tree02" / "tree.ply"
    print(full_points_file)

    voxel_map = VoxelMap(p(full_points_file))
    model = CubeForestModel()
    model.add_dataset(voxel_map)
    model.load()
    viz = model.predict()

    print("Load completed")
    meshes = []

    for (k, j, i, category) in viz:
        meshes.append(voxel(np.array([k, j, i]), category))

    o3d.visualization.draw_geometries(meshes)


if __name__ == "__main__":
    main()
