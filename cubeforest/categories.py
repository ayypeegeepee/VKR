from typing import NamedTuple


class Category(NamedTuple):
    weight: int
    min_points: int


Wood = Category(weight=1, min_points=50)
Something = Category(weight=1, min_points=100)