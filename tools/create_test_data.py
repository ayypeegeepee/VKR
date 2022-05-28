import numpy as np
import laspy
import pandas as pd

d = STEP = 15000000  # Ширина ребра куба


def process(filepath):
    print(f"Process: {filepath}")
    las = laspy.read(filepath)

    def get_points(k, j, i):
        keep1 = np.logical_and(las.X > d * i, las.Y > d * j)
        keep2 = np.logical_and(las.X < d * (i + 1), las.Y < d * (j + 1))
        more = np.logical_and(keep1, las.Z > d * k)
        less = np.logical_and(keep2, las.Z < d * (k + 1))
        condition = np.logical_and(more, less)
        points_x_i = las.X[condition]
        points_y_j = las.Y[condition]
        points_z_k = las.Z[condition]
        return points_x_i, points_y_j, points_z_k, len(points_x_i), len(points_y_j), len(points_z_k)

    min_x = las.X.min()
    min_y = las.Y.min()
    min_z = las.Z.min()

    max_x = las.X.max()
    max_y = las.Y.max()
    max_z = las.Z.max()

    las.X = las.X - min_x
    las.Y = las.Y - min_y
    las.Z = las.Z - min_z

    n_i = max_x // d + 1
    n_j = max_y // d + 1
    n_k = max_z // d + 1

    mtp = ((n_i + 1) * (1 + n_j) * (1 + n_k))
    buffer = np.empty((mtp, 5), dtype=object)

    cube_index = 0

    for n1 in range(n_k + 1):
        for n2 in range(n_j + 1):
            for n3 in range(n_i + 1):
                full_cloud = get_points(n1, n2, n3)
                buffer[cube_index][0] = n1
                buffer[cube_index][1] = n2
                buffer[cube_index][2] = n3
                buffer[cube_index][4] = full_cloud[4]
                if len(full_cloud[0]) != 0:
                    buffer[cube_index][3] = "Другое"
                else:
                    buffer[cube_index][3] = "Воздух"
                cube_index += 1

    df = pd.DataFrame(buffer, columns=['K', 'J', 'I', 'Type', 'length'])
    df.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
    return df


def main():
    print("Start working")
    # df_full = process("files\\single_tree_with_leaves.las")
    # df_tree = process("files\\broken_single_tree_no_leaves.las")

    df_full = process("..\\files\\full_4.5.22.las")
    df_tree = process("..\\files\\parted_4.5.22.las")

    full_points = df_full['length'].tolist()
    tree_points = df_tree['length'].tolist()
    buffer = np.empty((len(full_points), 5), dtype=object)

    for i in range(len(full_points)):
        if full_points[i] == 0:
            buffer[i][0] = 0
            buffer[i][1] = "Nothing"
        elif tree_points[i] == 0 and full_points[i] > 500:
            buffer[i][0] = 0
            buffer[i][1] = "Leaves"
        elif tree_points[i] > 100 and full_points[i] != 0:
            buffer[i][1] = "Tree"
        else:
            buffer[i][1] = "Nothing"

    file_length = len(df_full)
    k = df_full['K'].tolist()
    j = df_full['J'].tolist()
    i = df_full['I'].tolist()
    k_len = k[len(k) - 1]
    j_len = j[len(j) - 1]
    i_len = i[len(i) - 1]

    index = 0

    for k in range(k_len + 1):
        for j in range(j_len + 1):
            for i in range(i_len + 1):
                buffer[index][2] = k
                buffer[index][3] = j
                buffer[index][4] = i
                index += 1

    array = np.empty((k_len + 1, j_len + 1, i_len + 1), dtype=object)

    N = 0

    for k in range(k_len + 1):
        for j in range(j_len + 1):
            for i in range(i_len + 1):
                if buffer[N][1] == 'Nothing':
                    array[k][j][i] = 0
                    N += 1
                elif buffer[N][1] == 'Tree':
                    array[k][j][i] = 1
                    N += 1
                else:
                    array[k][j][i] = 2
                    N += 1

    viz = np.array(array)

    with open('..\\tmp\\viz_creation_2.npy', 'wb') as f:
        np.save(f, viz, allow_pickle=True)


if __name__ == '__main__':
    main()
