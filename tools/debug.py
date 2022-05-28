import pandas as pd
from pyntcloud import PyntCloud


pd.set_option('display.max_columns', None)


if __name__ == "__main__":
    single_tree_with_leaves = PyntCloud.from_file("..\\files\\single_tree_with_leaves.las")
    print("single_tree_with_leaves.las")
    print(single_tree_with_leaves.points.describe())

    single_tree_no_leaves = PyntCloud.from_file("..\\files\\single_tree_with_leaves.ply")
    print("broken_single_tree_no_leaves.las")
    print(single_tree_no_leaves.points.describe())

    pcd = o3d.io.read_point_cloud("..\\files\\single_tree_with_leaves.ply")