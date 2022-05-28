from pathlib import Path
from joblib import dump, load
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

import numpy as np


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

# def points_clustering(voxel_points):
#     aggregator = AffinityPropagation(preference=-50)
#     aggregator.fit(voxel_points[:, :-1], voxel_points[:, -1])
#     return aggregator.cluster_centers_indices_


def points_clustering(voxel_points):
    aggregator = KMeans(init="k-means++", n_clusters=2, n_init=10)
    aggregator.fit(voxel_points)
    return aggregator.cluster_centers_.flatten()


class CubeForestModel:
    def __init__(self):
        self._clf = None
        self._total_data_length = 0
        self._datasets = []

        # Медели, которые аппроксимируют точки в кубер
        # формат: (функция, число признаков, которые она генерит)
        self._features_extractor = (
            (points_clustering, 6),
            (points_linear_regression, 3),
            (points_pca, 2),
        )

        self._n_features = sum(m[1] for m in self._features_extractor)

    def save(self):
        weights_file = Path(__file__).parent / "weights" / "cube_forest_clf.joblib"
        dump(self._clf, weights_file)

    def load(self):
        weights_file = Path(__file__).parent / "weights" / "cube_forest_clf.joblib"
        self._clf = load(weights_file)

    def add_dataset(self, voxelmap):
        self._datasets.append(voxelmap)
        self._total_data_length += len(voxelmap.indexes[0])

    def _preprocess(self):
        pass

    def fit(self):
        X_calculated = np.empty((self._total_data_length, self._n_features))
        Y = np.empty((self._total_data_length, ))
        x_index = 0

        for dataset in self._datasets:
            for (p, category) in zip(dataset.get_data(), dataset.get_categories()):
                # Начинаем заполнять данные
                # Сначала по индексу берем точки принадлежащие кубу
                voxel_points = normalize(dataset.get_normalized_points_by_index(p[0], p[1], p[2]))

                # Высчитываем описательные метрики для всех точек в кубе и сохраняем значения
                X_calculated[x_index] = np.concatenate([f[0](voxel_points) for f in self._features_extractor])
                Y[x_index] = category
                x_index += 1

        X_train, X_test, y_train, y_test = train_test_split(
            X_calculated, Y, test_size=0.4, stratify=Y,
        )
        # Взял рандомный классификатор, как в статье
        self._clf = RandomForestClassifier(max_depth=3)
        # Учим
        self._clf.fit(X_train, y_train)
        # Пробуем
        predicted = self._clf.predict(X_test)
        # Сравниваем что получилось с тем, что должно получиться и пишем отчет
        print(metrics.classification_report(y_test, predicted))

    def predict(self):
        X_calculated = np.empty((self._total_data_length, self._n_features))
        x_index = 0

        dataset = self._datasets[0]
        for p in dataset.get_data():
            voxel_points = normalize(dataset.get_normalized_points_by_index(p[0], p[1], p[2]))
            X_calculated[x_index] = np.concatenate([f[0](voxel_points) for f in self._features_extractor])
            x_index += 1

        predicted = self._clf.predict(X_calculated)
        return np.c_[dataset.get_data(), predicted]
