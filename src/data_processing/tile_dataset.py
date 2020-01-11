import numpy as np
from typing import Union, List, Tuple, Dict, Any
from math import floor, ceil
from detectron2.structures import BoxMode
import itertools


def min_max(arr: np.ndarray) -> Tuple[float, float]:
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


class TiledDataset:
    def __init__(self, shape_record: Dict['str', Any], width, height=None, w_overlay=0, h_overlay=None):
        self.annotations = shape_record['annotations']
        self.filename = shape_record['filename']
        self.x_max, self.y_max = shape_record['width'], shape_record['height']
        self.id_0 = shape_record['image_id']
        self.tiled_record: List[Dict] = []
        self.width, self.height = width, height
        if height is None:
            self.height = width
        if h_overlay is None:
            self.h_overlay = w_overlay
        self.dx, self.dy = (width - w_overlay), (height - h_overlay)
        self.num_tiles = floor(self.x_max / self.dx) * floor(self.y_max / self.dy)

    def tile_ortho(self):
        pass

    def tile_polygons(self) -> List[Dict[str, Union[str, int, float]]]:

        for tile in range(self.num_tiles):
            self.tiled_record.append({'file_name': f'{self.filename}_{tile}',
                                      'image_id': tile + self.id_0,
                                      'width': self.width, 'height': self.height, 'annotations': []})

            for annotation in self.annotations:
                x_min, y_min, x_max, y_max = annotation['bbox']
                x_pos, y_pos = floor(x_min / self.dx), floor(y_min / self.dy)
                x_lower_grid_bound, y_lower_grid_bound = self.dx * x_pos, self.dy * y_pos
                if box_in_box(annotation['bbox'], [x_lower_grid_bound, y_lower_grid_bound, self.width, self.height]):
                    # id: (x_pos + 1)(y_pos + 1) - 1
                    self.annotations[(x_pos + 1) * (y_pos + 1) - 1]['annotations'].append(
                        create_annotation(
                            poly=annotation['segmentation'],
                            bbox=annotation['bbox'],
                            rescale_corner=(x_lower_grid_bound, y_lower_grid_bound),
                            is_crowd=annotation['iscrowd'],
                            category_id=annotation['category_id']
                        )
                    )
                elif box_in_box(annotation['bbox'],
                                [x_lower_grid_bound - self.width, y_lower_grid_bound, self.width, self.height]):
                    # id: (x_pos)(y_pos + 1) - 1
                    self.annotations[x_pos * (y_pos + 1) - 1]['annotations'].append(
                        create_annotation(
                            poly=annotation['segmentation'],
                            bbox=annotation['bbox'],
                            rescale_corner=(x_lower_grid_bound - self.width, y_lower_grid_bound),
                            is_crowd=annotation['iscrowd'],
                            category_id=annotation['category_id']
                        )
                    )
                elif box_in_box(annotation['bbox'],
                                [x_lower_grid_bound - self.width, y_lower_grid_bound - self.height, self.width,
                                 self.height]):
                    # id: (x_pos)(y_pos) - 1
                    self.annotations[x_pos * y_pos - 1]['annotations'].append(
                        create_annotation(
                            poly=annotation['segmentation'],
                            bbox=annotation['bbox'],
                            rescale_corner=(x_lower_grid_bound - self.width, y_lower_grid_bound - self.height),
                            is_crowd=annotation['iscrowd'],
                            category_id=annotation['category_id']
                        )
                    )
                elif box_in_box(annotation['bbox'],
                                [x_lower_grid_bound, y_lower_grid_bound - self.height, self.width, self.height]):
                    # id: (x_pos + 1)(y_pos) - 1
                    self.annotations[(x_pos + 1) * y_pos - 1]['annotations'].append(
                        create_annotation(
                            poly=annotation['segmentation'],
                            bbox=annotation['bbox'],
                            rescale_corner=(x_lower_grid_bound, y_lower_grid_bound - self.height),
                            is_crowd=annotation['iscrowd'],
                            category_id=annotation['category_id']
                        )
                    )

            return self.tiled_record


def create_annotation(poly, bbox, rescale_corner, is_crowd, category_id):
    return {
        "bbox": [bbox[0] - rescale_corner[0], bbox[1], rescale_corner[1], bbox[2] - rescale_corner[0],
                 bbox[3] - rescale_corner[1]],
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [list(itertools.chain.from_iterable(map(lambda x: shift(x, rescale_corner), poly)))],
        "category_id": category_id,
        # "category_id": rec.segClass, # classification enabled, we can force each segment to a
        # single tree class if needed
        "iscrowd": is_crowd
        # iscrowd groups individual objects of the same kind into a single segment. For tree segmentation,
        # we want to isolate trees
    }


def shift(val, corner):
    return val[0] - corner[0], val[1] - corner[1]


def box_in_box(contained, container):
    """
    :param container: Rectangle [x min, y min, w, h]
    :param contained: Rectangle [x min, y min, x max, y max]
    :return: T/F Is contained in container?
    """
    if min(container[:2]) < 0:
        return False
    if contained[0] >= container[0] and contained[1] >= container[1] and contained[2] <= container[0] + container[2] \
            and contained[3] <= container[1] + container[3]:
        return True
    return False
