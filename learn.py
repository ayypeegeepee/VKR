import numpy as np
import open3d as o3d
from pathlib import Path

from cubeforest import CubeForestModel, VoxelMap
from cubeforest.categories import Wood

PC_FILE_EXTENSION = ".ply"


def p(points_file):
    input_cloud = o3d.io.read_point_cloud(str(points_file))
    return np.asarray(input_cloud.points)


def main():
    data_dir = Path("data")
    model = CubeForestModel()

    dirs = [tree_dir for tree_dir in data_dir.iterdir() if tree_dir.name in ["tree02"]]

    for tree_dir in dirs:
        full_points_file = tree_dir / "tree.ply"
        wood_points_file = tree_dir / "wood.ply"
        processed_npy_file = tree_dir / "voxelized.npy"
        print(full_points_file)
        print(wood_points_file)

        voxel_map = VoxelMap(p(full_points_file))
        voxel_map.add_category(p(wood_points_file), Wood)

        voxel_map.save(processed_npy_file)
        # voxel_map.load(processed_npy_file)

        model.add_dataset(voxel_map)

    model.fit()
    model.save()


if __name__ == "__main__":
    main()
