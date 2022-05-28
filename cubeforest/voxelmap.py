import numpy as np
from numba.experimental import jitclass
from numba import int8, float32
from numba import jit, prange
from sklearn.preprocessing import normalize

from .settings import VOXEL_LENGTH
from .categories import Category, Something


@jit(nopython=True, fastmath=True)
def _numba_slice_point_indexes(points, k, j, i, min_x, min_y, min_z):
    return (min_x + VOXEL_LENGTH * i < points[:, 0]) & (points[:, 0] < min_x + VOXEL_LENGTH * (i + 1)) & \
           (min_y + VOXEL_LENGTH * j < points[:, 1]) & (points[:, 1] < min_y + VOXEL_LENGTH * (j + 1)) & \
           (min_z + VOXEL_LENGTH * k < points[:, 2]) & (points[:, 2] < min_z + VOXEL_LENGTH * (k + 1))


@jit(nopython=True, fastmath=True)
def _numba_slice_points_len(points, k, j, i, min_x, min_y, min_z):
    return _numba_slice_point_indexes(points, k, j, i, min_x, min_y, min_z).sum()


@jit(nopython=True, fastmath=True)
def _numba_slice_points(points, k, j, i, min_x, min_y, min_z):
    return points[_numba_slice_point_indexes(points, k, j, i, min_x, min_y, min_z)]


@jit(nopython=True, parallel=True, fastmath=True)
def _numba_add_category(voxel_map, shape, points, min_x, min_y, min_z, min_points, weight):
    for k in prange(shape[0]):
        for j in prange(shape[1]):
            for i in prange(shape[2]):
                point_count = _numba_slice_points_len(points, k, j, i, min_x, min_y, min_z)
                if point_count >= min_points:
                    voxel_map[k][j][i] += weight


class VoxelMap:
    """Облако точек с дополнительными операциями"""

    def __init__(self, points):
        self.points = points
        self.min_x = self.points[:, 0].min()
        self.min_y = self.points[:, 1].min()
        self.min_z = self.points[:, 2].min()
        self.max_x = self.points[:, 0].max()
        self.max_y = self.points[:, 1].max()
        self.max_z = self.points[:, 2].max()

        self.shape = (
            int((self.max_z - self.min_z) // VOXEL_LENGTH + 1),
            int((self.max_y - self.min_y) // VOXEL_LENGTH + 1),
            int((self.max_x - self.min_x) // VOXEL_LENGTH + 1),
        )

        self.indexes = None
        self.voxel_map = np.zeros(self.shape, dtype=np.int8)
        self.add_category(self.points, Something)

    def _slice_point_indexes(self, points, k, j, i):
        """Возвращает индексы точек, принадлежащих кубу с началом c индексом K,J,I"""
        return (self.min_x + VOXEL_LENGTH * i < points[:, 0]) & (points[:, 0] < self.min_x + VOXEL_LENGTH * (i + 1)) & \
               (self.min_y + VOXEL_LENGTH * j < points[:, 1]) & (points[:, 1] < self.min_y + VOXEL_LENGTH * (j + 1)) & \
               (self.min_z + VOXEL_LENGTH * k < points[:, 2]) & (points[:, 2] < self.min_z + VOXEL_LENGTH * (k + 1))

    def save(self, filename):
        with open(filename, 'wb') as f:
            np.save(f, self.voxel_map, allow_pickle=True)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.voxel_map = np.load(f, allow_pickle=True)

    def add_category(self, points, category: Category):
        _numba_add_category(self.voxel_map, self.shape, points, self.min_x, self.min_y, self.min_z, category.min_points,
                            category.weight)
        self.indexes = self.voxel_map.nonzero()

    def get_categories(self):
        return self.voxel_map[self.indexes]

    def get_data(self):
        # и из самих индексом собираем двухмерный массив с k.j.i, y
        return np.stack(self.indexes, axis=1)

    def get_normalized_points_by_index(self, k, j, i):
        return _numba_slice_points(self.points, k, j, i, self.min_x, self.min_y, self.min_z)
