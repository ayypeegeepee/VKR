from numba import jit
import numpy as np
import open3d as o3d
import time


@jit(nopython=True, parallel=True)
def voxelize(pc, voxel_length, min_x, min_y, min_z, shape):
    # point_limit = 30 Было 30, но дереово портится
    point_limit = 10  # Минимум 3 иначе сломается PCA

    result = np.zeros(shape, dtype=np.int8)
    for k in range(shape[0]):
        for j in range(shape[1]):
            for i in range(shape[2]):
                point_count = (
                        (min_x + voxel_length * i < pc[:, 0]) & (pc[:, 0] < min_x + voxel_length * (i + 1)) &
                        (min_y + voxel_length * j < pc[:, 1]) & (pc[:, 1] < min_y + voxel_length * (j + 1)) &
                        (min_z + voxel_length * k < pc[:, 2]) & (pc[:, 2] < min_z + voxel_length * (k + 1))
                ).sum()
                if point_count >= point_limit:
                    result[k][j][i] = 1

    return result


def main():
    voxel_length = 0.5

    # pc_tree_with_leaves = laspy.read("..\\files\\single_tree_with_leaves.las")
    # Если используем laspy, то pc.xyz - pc.points.offset

    input_cloud = o3d.io.read_point_cloud("files\\single_tree_with_leaves.ply")
    pc = np.asarray(input_cloud.points)

    min_x, max_x = pc[:, 0].min(), pc[:, 0].max()
    print("X",  min_x, max_x)
    min_y, max_y = pc[:, 1].min(), pc[:, 1].max()
    print("X",  min_y, max_y)
    min_z, max_z = pc[:, 2].min(), pc[:, 2].max()
    print("X",  min_z, max_z)

    shape = (
        int((max_z - min_z) // voxel_length + 1),
        int((max_y - min_y) // voxel_length + 1),
        int((max_x - min_x) // voxel_length + 1),
    )
    print(shape)
    start = time.time()
    full_tree_voxels = voxelize(pc, voxel_length, min_x, min_y, min_z, shape)
    print(f"Finish working with full_tree_voxels: {time.time() - start}")

    input_cloud = o3d.io.read_point_cloud("files\\broken_single_tree_no_leaves.ply")
    pc = np.asarray(input_cloud.points)
    start = time.time()
    only_tree_voxels = voxelize(pc, voxel_length, min_x, min_y, min_z, shape)
    print(f"Finish working with broken_single_tree_no_leaves: {time.time() - start}")
    result = full_tree_voxels + only_tree_voxels

    with open('tmp\\voxelized.npy', 'wb') as f:
        np.save(f, result, allow_pickle=True)


if __name__ == "__main__":
    main()
