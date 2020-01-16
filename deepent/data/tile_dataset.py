import glob
import itertools
import json
from math import floor, ceil
from shutil import copyfile
from typing import Union, List, Tuple, Dict, Any

import cv2
import numpy as np
from skimage.io import imread
from tqdm import tqdm


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
        self.filename = shape_record['file_name']
        self.x_max, self.y_max = shape_record['width'], shape_record['height']
        self.id_0 = shape_record['image_id']
        self.tiled_record: List[Dict] = []

        self.width, self.height = width, height
        self.w_overlay, self.h_overlay = w_overlay, h_overlay
        if self.height is None:
            self.height = width
        if self.h_overlay is None:
            self.h_overlay = w_overlay

        self.dx, self.dy = (width - w_overlay), (height - h_overlay)

        self.x_tiles = ceil(self.x_max / self.dx)
        self.num_tiles = self.x_tiles * ceil(self.y_max / self.dy)

    def tile_ortho(self):
        file_name, data_set_name = (lambda x: (x[-1][:-4], x[-2]))(self.filename.split('/'))
        output_dir = f'tiled_datasets/{data_set_name}'
        img = imread(self.filename)
        print('Ortho Read')
        idx = self.id_0
        x_max, y_max = img.shape[:2]
        for x in range(0, x_max, self.dx):
            for y in range(0, y_max, self.dy):
                x_c = x + self.width if x + self.width <= x_max else x_max
                y_c = y + self.height if y + self.height <= y_max else y_max

                cv2.imwrite(f'{output_dir}/{file_name}_{idx}.png', img[x:x_c, y:y_c])
                idx += 1
                if idx % 100 == 0:
                    print(f'Tile {idx}/{self.num_tiles} written')

    def tile_polygons(self) -> List[Dict[str, Union[str, int, float]]]:
        for tile in range(self.num_tiles):
            self.tiled_record.append({'file_name': f'{self.filename[:-4]}_{tile}.png',
                                      'image_id': tile + self.id_0,
                                      'width': self.width, 'height': self.height, 'annotations': []})

        print(f'Tiling Polygons into {self.num_tiles} Tiles.')
        for annotation in tqdm(self.annotations):
            x_min, y_min, x_max, y_max = annotation['bbox']
            x_pos, y_pos = (x_min // self.dx), (y_min // self.dy)

            for x_shift in (0, self.dx):
                for y_shift in (0, self.dy):
                    x, y = self.dx * x_pos - x_shift, self.dy * y_pos - y_shift
                    x_c = x + self.width if x + self.width < self.x_max else self.x_max
                    y_c = y + self.height if y + self.height < self.y_max else self.y_max

                    if box_in_box(annotation['bbox'], [x, y, x_c, y_c]):
                        x_pos_c, y_pos_c = x_pos - if_non_zero(x_shift), y_pos - if_non_zero(y_shift)
                        self.tiled_record[int(self.x_tiles * y_pos_c + x_pos_c)]['annotations'].append(
                            create_annotation(
                                poly=annotation['segmentation'],
                                bbox=annotation['bbox'],
                                rescale_corner=(x, y),
                                is_crowd=annotation['iscrowd'],
                                category_id=annotation['category_id']
                            )
                        )

        return self.tiled_record


def create_annotation(poly, bbox, rescale_corner, is_crowd, category_id):
    return {
        "bbox": [bbox[0] - rescale_corner[0], bbox[1] - rescale_corner[1], bbox[2] - rescale_corner[0],
                 bbox[3] - rescale_corner[1]],
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
    :param container: Rectangle [x min, y min, x_max, y_max]
    :param contained: Rectangle [x min, y min, x max, y max]
    :return: T/F Is contained in container?
    """
    if min(container[:2]) < 0:
        return False
    if contained[0] >= container[0] and contained[1] >= container[1] and contained[2] <= container[2] \
            and contained[3] <= container[3]:
        return True
    return False


def if_non_zero(x):
    if x == 0:
        return 0
    return 1


def partition_tiles(train_ratio=.8):
    datasets = glob.glob('tiled_datasets/*')
    print(datasets)
    datasets.sort()
    for dataset in datasets:
        dataset_name = dataset.split('/')[-1]
        print(f'Partioning {dataset_name}')
        with open(f'{dataset}/segs.json', 'r') as f:
            segs = json.load(f)

        train, test = [], []
        train_batch_size = floor(100 * train_ratio)
        id_0 = int(sorted(glob.glob(f'{dataset}/*.png'))[0].split('/')[-1].split('.')[-2].split('_')[-1])

        for i, seg in tqdm(enumerate(segs)):
            image_name = seg['file_name'].split('/')[-1]
            img_num = int(image_name.split('_')[-1][:-4])
            image_name = '_'.join(image_name.split('_')[:-1]) + f'_{id_0 + img_num}.png'
            assert id_0 + img_num == seg['image_id']
            if i % 100 <= train_batch_size:
                # Training data
                new_name = f'FYBRData/train/{dataset_name}/{image_name}'
                copyfile(f'tiled_datasets/{dataset_name}/{image_name}', new_name)
                seg['file_name'] = new_name
                train.append(seg)
            else:
                # Test Data
                new_name = f'FYBRData/test/{dataset_name}/{image_name}'
                copyfile(f'tiled_datasets/{dataset_name}/{image_name}', new_name)
                seg['file_name'] = new_name
                test.append(seg)

        with open(f'FYBRData/train/{dataset_name}/segs.json', 'w') as f:
            f.write(json.dumps(train))

        with open(f'FYBRData/test/{dataset_name}/segs.json', 'w') as f:
            f.write(json.dumps(test))


def fix_polygon_tail(polygon):
    first = polygon[0]
    new_poly = []
    for i, p in enumerate(polygon):
        new_poly.append(p)
        if i > 0 and p == first:
            break
    return new_poly


if __name__ == '__main__':
    partition_tiles()