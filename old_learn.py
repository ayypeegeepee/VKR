import numpy as np
import open3d as o3d

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def read_cloud(filepath):
    with open(filepath, 'rb') as f:
        viz = np.load(f, allow_pickle=True)
    return viz


# Скопировал функцию получения точек по индексу
def get_points(pc, k, j, i, min_x, min_y, min_z):
    voxel_length = 0.5
    return pc[
        (min_x + voxel_length * i < pc[:, 0]) & (pc[:, 0] < min_x + voxel_length * (i + 1)) &
        (min_y + voxel_length * j < pc[:, 1]) & (pc[:, 1] < min_y + voxel_length * (j + 1)) &
        (min_z + voxel_length * k < pc[:, 2]) & (pc[:, 2] < min_z + voxel_length * (k + 1))
    ]


def points_pca(voxel_points):
    # Пытаемся уменьшить размерность области, чтобы оставить суть
    n_components = 3
    aggregator = PCA(n_components=n_components)
    aggregator.fit(voxel_points)
    return aggregator.explained_variance_ratio_


# берем линейную регрессию и пытаемся сформировать уравнение описывающее плоскостть через все точки, типо форму
def points_linear_regression(voxel_points):
    aggregator = linear_model.LinearRegression()
    aggregator.fit(voxel_points[:, :-1], voxel_points[:, -1])
    return aggregator.coef_


if __name__ == "__main__":
    # Берем сохраненный ранее массив с индексами кубов и классами
    voxel_classes = read_cloud("tmp\\voxelized.npy")

    # И само облако точек
    input_cloud = o3d.io.read_point_cloud("files\\single_tree_with_leaves.ply")
    all_points = np.asarray(input_cloud.points)

    # Снова высчитываем границы, чтобы корректно получать точки по индексу
    min_x, min_y, min_z = all_points[:, 0].min(), all_points[:, 1].min(), all_points[:, 2].min()
    print("Min:",  min_x, min_y, min_z)

    # Берем все индексы кубов, в которх не ноль
    indexes = voxel_classes.nonzero()
    # Создаем вектор категорий по полученным индексам
    Y = voxel_classes[indexes]
    # и из самих индексом собираем двухмерный массив с k.j.i
    X = np.stack(indexes, axis=1)  # Получаем в формате z,y,x, класс (1/2)

    # Сюда будем писать посчитанные значения
    X_calculated = np.empty((len(Y), 5))  # Хардкод, 2 + 3 признака из points_linear_regression + points_pca
    for (i, p) in enumerate(X):
        # Начинаем заполнять данные
        # Сначала по индексу берем точки принадлежащие кубу
        voxel_points = get_points(all_points, p[0], p[1], p[2], min_x, min_y, min_z)
        # Нормализзуем, чтобы значения были от 0 до 1, те не зависило от положения на дереве
        voxel_points = normalize(voxel_points)
        # Высчитываем описательные метрики для всех точек в кубе и сохраняем значения
        X_calculated[i] = np.append(points_linear_regression(voxel_points), points_pca(voxel_points))

    # Полученные метррики делим на обучающие и проверочные выборки
    # test_size - это сколкьо процентов выборки отправляем на тест
    # stratify - если указать классы, то деление идет равномерно, даже с учетом, что в одном классе много больше точек
    X_train, X_test, y_train, y_test = train_test_split(
        X_calculated, Y, test_size=0.1, stratify=Y,
    )

    # Взял рандомный классификатор, как в статье
    clf = RandomForestClassifier(max_depth=2, random_state=0)

    # Учим
    clf.fit(X_train, y_train)

    # Пробуем
    predicted = clf.predict(X_test)

    # Сравниваем что получилось с тем, что должно получиться и пишем отчет
    print(metrics.classification_report(y_test, predicted))

