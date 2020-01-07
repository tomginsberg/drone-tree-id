import numpy as np
from typing import Union, List, Tuple


def min_max(arr: np.ndarray[float]) -> Tuple[float, float]:
    """
    For some reason this is not a numpy function
    :param arr:
    :return:
    """
    return np.min(arr), np.max(arr)


def polygon_in_rectangle(poly: np.ndarray, rect: Union[np.ndarray, List[Tuple[float, float]]]) -> bool:
    """
    Check if a polygon is fully contained inside a rectangle
    Assumes a top right (0, 0) origin
    :param poly: List of 2D points
    :param rect: a rectangle defined by [(top_right_x, top_right_y), (width, height)]
    :return: T/F is poly \in rect
    """
    p_x, p_y = poly.transpose()
    x_min, x_max = min_max(p_x)
    y_min, y_max = min_max(p_y)
    if x_min > rect[0][0] and y_min > rect[0][1] and x_max < rect[0][0] + rect[1][0] and y_max < rect[0][1] \
            + rect[1][1]:
        return True
    return False


def rect_corners(rect: Union[np.ndarray, List[Tuple[float, float]]]) -> np.ndarray:
    x, y = list(rect[0])
    w, h = list(rect[1])
    return np.array([[x, y], [x + w, y], [x + w, y + w], [x, y + w], [x, y]]).transpose()


class TiledPolygons:
    def __init__(self, polygons: List[List[Tuple[float, float]]], domain: Tuple[float, float]):
        self.polygons = polygons

    def tile(self, width, height=None, w_overlay=0, h_overlay=None) -> np.ndarray:
        if height is None:
            height = width
        if h_overlay is None:
            h_overlay = w_overlay
        # placeholder
        return np.array([0])
