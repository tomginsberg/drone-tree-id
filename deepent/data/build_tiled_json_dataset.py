import itertools
import json
import os
import shutil
from glob import glob
from math import ceil
from typing import List, Dict, Union, Any
from typing import Tuple

import cv2
import numpy as np
import rasterio
import shapefile
from skimage.io import imread
from tqdm import tqdm

np.random.seed(42)


# TODO: Complete Docstrings
class DataTiler:
    def __init__(self, input_dir: str = 'datasets', output_dir: str = 'test_dataset', tile_width: int = 600,
                 tile_height: int = 600, horizontal_overlay: int = 300, vertical_overlay: int = 300,
                 cleanup_on_init: bool = False):
        """
        :param input_dir:
        :param output_dir:
        :param tile_width:
        :param tile_height:
        :param horizontal_overlay:
        :param vertical_overlay:
        :param cleanup_on_init:
        """
        self.input_dir, self.output_dir = input_dir, output_dir

        self.tile_width, self.tile_height = tile_width, tile_height
        self.horizontal_overlay, self.vertical_overlay = horizontal_overlay, vertical_overlay

        self.dataset_input_paths = glob(os.path.join(input_dir, '*'))
        self.dataset_names = [os.path.basename(path) for path in self.dataset_input_paths]

        self.dx, self.dy = (tile_width - horizontal_overlay), (tile_height - vertical_overlay)
        self.classes = {}

        if cleanup_on_init:
            try:
                self.cleanup()
            except FileNotFoundError:
                print('No directory to cleanup. Proceeding')

    def _tile_segments(self, dataset_directory: str, ortho_name: str, img: np.ndarray) -> List[List[Dict[str, any]]]:
        """

        :param dataset_directory:
        :param ortho_name:
        :param img:
        :return: annotations: List[List[Dict]]
        """
        # Use rasterio to compute geometric transformation parameters
        ds = rasterio.open(ortho_name, 'r')
        ulx, xres, _, uly, _, yres = ds.get_transform()

        # Size of Ortho in geometric scale
        orthox = xres * ds.width
        orthoy = yres * ds.height

        img_h, img_w = img.shape[:2]

        # Locate ShapeFile in datasets/dataset/Segments/ directory
        shapefile_name = glob(os.path.join(dataset_directory, 'Segments', '*'))[0][:-4]

        print(f'Importing Shapefile {shapefile_name}')
        shpf = shapefile.Reader(shapefile_name)

        shape_recs = shpf.shapeRecords()
        bbox = shpf.bbox

        minx = bbox[0]
        maxy = bbox[3]

        # Offset of Shapefile from top left corner of Ortho [0, width/height] in geometric scale
        offsetx = minx - ulx
        offsety = maxy - uly

        # Scale is ratio of image pixel width/height to geometric width/height in raster
        scalex = img_w / orthox
        scaley = img_h / orthoy
        rescale_x = lambda x: (x - minx + offsetx) * scalex
        rescale_y = lambda y: (y - maxy + offsety) * scaley

        # Compute the number of image tiles and initialize annotation
        x_tiles = ceil(img_w / self.dx)
        num_tiles = x_tiles * ceil(img_h / self.dy)
        annotations = [[] for _ in range(num_tiles)]

        print(f'Tiling Shapes for {os.path.basename(dataset_directory)}')
        for shape_rec in tqdm(shape_recs):
            shp = shape_rec.shape
            if shp.shapeType == 5:  # 5 - polygon

                class_name = shape_rec.record.segClass
                # Update class dict
                if class_name not in self.classes:
                    self.classes[class_name] = len(self.classes)

                # transform polygon and bounding into image coordinate system
                rescaled_poly = [[rescale_x(x), rescale_y(y)] for x, y in fix_polygon_tail(shp.points)]

                shape_bbox = [rescale_x(shp.bbox[0]), rescale_y(shp.bbox[3]),
                              rescale_x(shp.bbox[2]), rescale_y(shp.bbox[1])]

                x_min, y_min, x_max, y_max = shape_bbox
                x_pos, y_pos = (x_min // self.dx), (y_min // self.dy)

                # Compute tiles that shape belongs to
                for y_shift in (0, self.dy):
                    for x_shift in (0, self.dx):
                        x, y = self.dx * x_pos - x_shift, self.dy * y_pos - y_shift
                        x_c = x + self.tile_width if x + self.tile_width < img_w else img_w
                        y_c = y + self.tile_height if y + self.tile_height < img_h else img_h

                        if box_in_box(shape_bbox, [x, y, x_c, y_c]):
                            x_pos_c, y_pos_c = x_pos - if_non_zero(x_shift), y_pos - if_non_zero(y_shift)
                            annotations[int(x_tiles * y_pos_c + x_pos_c)].append(
                                create_annotation(
                                    poly=rescaled_poly,
                                    bbox=shape_bbox,
                                    rescale_corner=(x, y),
                                    is_crowd=0,
                                    category_id=self.classes[class_name]
                                )
                            )
        return annotations

    def tile_dataset(self, tile_filtering_function=None, train_limit: int = 80):
        """

        :param tile_filtering_function:
        :param train_limit:
        :raises FileNotFoundError: if a single file matching *ortho.tif cannot be found in the dataset path
        """
        self.build_output_dir_structure()
        for dataset_directory, dataset_name in zip(self.dataset_input_paths, self.dataset_names):
            ortho_name = glob(os.path.join(dataset_directory, '*ortho.tif'))
            if len(ortho_name) != 1:
                raise FileNotFoundError('Either 0 or > 1 files matching *ortho.tif found.')
            ortho_name = ortho_name[0]

            chm_name = glob(os.path.join(dataset_directory, '*up_CHM.tif'))
            if len(chm_name) != 1:
                raise FileNotFoundError('Either 0 or > 1 files matching *up_CHM.tif found.')
            chm_name = chm_name[0]

            print(f'Reading Ortho: {ortho_name}')
            ortho = imread(ortho_name)

            print(f'Reading Ortho: {chm_name}')
            chm = imread(chm_name)
            print('Normalizing CHM')
            chm = np.where(chm == -np.inf, 0, chm)
            chm = (255 * (chm - np.min(chm)) / np.max(chm)).astype(ortho.dtype)

            assert (chm.shape[:2] == ortho.shape[:2]), 'Ortho and CHM are different shapes'
            assert ortho.shape[-1] == 4, 'Ortho should contain a 4th channel'

            annotations = self._tile_segments(dataset_directory, ortho_name, ortho)

            img_h, img_w = ortho.shape[:2]
            tile_number, train_id, test_id = 0, 0, 0
            train_record, test_record = [], []
            num_tiles = ceil(img_w / self.dx) * ceil(img_h / self.dy)

            for y in range(0, img_h, self.dy):
                for x in range(0, img_w, self.dx):
                    train_test = np.random.randint(0, 100)
                    x_c = x + self.tile_width if x + self.tile_width <= img_w else img_w
                    y_c = y + self.tile_height if y + self.tile_height <= img_h else img_h
                    if train_test <= train_limit:
                        image_output_dir = os.path.join(self.output_dir, 'train', dataset_name, f'tile_{train_id}.png')
                        train_id += 1
                        train_record.append(
                            {'file_name': image_output_dir, 'image_id': tile_number, 'width': (x_c - x),
                             'height': (y_c - y),
                             'annotations': annotations[tile_number]})
                    else:
                        image_output_dir = os.path.join(self.output_dir, 'test', dataset_name, f'tile_{test_id}.png')
                        test_id += 1
                        test_record.append(
                            {'file_name': image_output_dir, 'image_id': tile_number, 'width': (x_c - x),
                             'height': (y_c - y),
                             'annotations': annotations[tile_number]})

                    tile = ortho[y:y_c, x:x_c]
                    tile[:, :, -1] = chm[y:y_c, x:x_c]

                    cv2.imwrite(image_output_dir, tile)
                    tile_number += 1
                    if tile_number % 200 == 0:
                        print(f'Tile # {tile_number} of {num_tiles} created for {dataset_name}')

            for train_test, rec in zip(['train', 'test'], [train_record, test_record]):
                with open(os.path.join(self.output_dir, train_test, dataset_name, 'segs.json'), 'w') as f:
                    f.write(json.dumps(rec))

        print(f'Writing Classes f{self.classes}')
        with open(os.path.join(self.output_dir, 'classes.json'), 'w') as f:
            f.write(json.dumps(self.classes))

    def build_output_dir_structure(self):
        """
        Constructs the file structure
            output_dir
            |__train
            | |__dataset1
            | |__dataset2
            | |__dataset3
            | |__...
            |__test
            | |__dataset1
            | |__dataset2
            | |__dataset3
            | |__...
        :raises FileExistsError: if self.output_dir already exists
        """
        try:
            os.mkdir(self.output_dir)
        except FileExistsError:
            print(
                f'Output Directory: "{self.output_dir}" already exists. '
                f'Please clear it manually, with DataTiler::cleanup() or specify a different directory.')
            raise FileExistsError
        os.chdir(self.output_dir)

        for dir_name in ['train', 'test']:
            os.mkdir(dir_name)
            os.chdir(dir_name)
            for dataset_name in self.dataset_names:
                os.mkdir(dataset_name)
            os.chdir('..')
        os.chdir('..')

    def cleanup(self):
        """
        Removes self.output_dir and all containing files
        """
        print(f'Cleaning directory: {self.output_dir}')
        shutil.rmtree(self.output_dir)


def fix_polygon_tail(polygon: List[List[float]]) -> List[List[float]]:
    """
    :param polygon:
    :return:
    """
    first = polygon[0]
    new_poly = []
    for i, p in enumerate(polygon):
        new_poly.append(p)
        if i > 0 and p == first:
            break
    return new_poly


def if_non_zero(x: Union[int, float]) -> int:
    """
    :param x: any int/float
    :return: 0 if x == 0 else 1
    """
    if x == 0:
        return 0
    return 1


def box_in_box(contained: List[float], container: List[float]) -> bool:
    """
    :param container: Rectangle [x min, y min, x_max, y_max]
    :param contained: Rectangle [x min, y min, x max, y max]
    :return: T/F Is contained a subset container?
    """
    if min(container[:2]) < 0:
        return False
    if contained[0] >= container[0] and contained[1] >= container[1] and contained[2] <= container[2] \
            and contained[3] <= container[3]:
        return True
    return False


def shift(val: List[float], corner: Tuple[float, float]) -> Tuple[float, float]:
    """

    :param val:
    :param corner:
    :return:
    """
    return val[0] - corner[0], val[1] - corner[1]


def create_annotation(poly: List[List[float]], bbox: List[float], rescale_corner: Tuple[float, float],
                      is_crowd: int, category_id: int) -> Dict[str, Any]:
    """

    :param poly:
    :param bbox:
    :param rescale_corner:
    :param is_crowd:
    :param category_id:
    :return:
    """
    return {
        "bbox": [bbox[0] - rescale_corner[0], bbox[1] - rescale_corner[1], bbox[2] - rescale_corner[0],
                 bbox[3] - rescale_corner[1]],
        "segmentation": [list(itertools.chain.from_iterable(map(lambda x: shift(x, rescale_corner), poly)))],
        "category_id": category_id,
        "iscrowd": is_crowd
    }


if __name__ == '__main__':
    # dt = DataTiler('datasets', 'RGBD-Tree-Segs', cleanup_on_init=False)
    dt = DataTiler('/home/ubuntu/datasets', '/home/ubuntu/RGBD-Tree-Segs', cleanup_on_init=False)
    dt.tile_dataset()
    # dt.cleanup()
