import itertools
import json
import os
import shutil
from glob import glob
from math import ceil
from typing import List, Dict, Any
from typing import Tuple, Union

import cv2
import numpy as np
import rasterio
import shapefile
from skimage.io import imread
from tqdm import tqdm

np.random.seed(42)


class DataTiler:
    def __init__(self, input_dir: str, output_dir: str, tile_width: int = 640,
                 tile_height: int = 640, horizontal_overlay: int = 320, vertical_overlay: int = 320,
                 cleanup_on_init: bool = False, compute_means=False, dataset_regex: Union[str, List[str]] = '*'):
        """
        :param input_dir:
        :param output_dir:
        :param tile_width:
        :param tile_height:
        :param horizontal_overlay:
        :param vertical_overlay:
        :param cleanup_on_init:
        """
        self.input_dir, self.output_dir = os.path.realpath(input_dir), os.path.realpath(output_dir)

        self.tile_width, self.tile_height = tile_width, tile_height
        self.horizontal_overlay, self.vertical_overlay = horizontal_overlay, vertical_overlay

        if len(glob(os.path.join(input_dir, '*.tif'))) > 0:
            self.dataset_input_paths = [self.input_dir]
            self.dataset_names = [os.path.basename(path) for path in self.dataset_input_paths]
            print(f'Running predictions on {self.dataset_names[0]}')
        else:
            if isinstance(dataset_regex, str):
                dataset_regex = [dataset_regex]
            self.dataset_input_paths = [x for x in set().union(
                *[glob(os.path.join(self.input_dir, reg)) for reg in dataset_regex]) if 'tmp' not in x]
            self.dataset_names = [os.path.basename(path) for path in self.dataset_input_paths]
            print('Paths:\n', self.dataset_input_paths, '\nNames:\n', self.dataset_names)

            self.dx, self.dy = (tile_width - horizontal_overlay), (tile_height - vertical_overlay)
            self.classes = {}
            self.compute_means = compute_means
            if compute_means:
                self.ortho_means = []
            self.chm_means = []

            if cleanup_on_init:
                try:
                    self.cleanup()
                except FileNotFoundError:
                    print('No directory to cleanup. Proceeding')

    def _tile_segments(self, dataset_directory: str, ds: rasterio.DatasetReader,
                       bbox_filtering_function) -> List[List[Dict[str, any]]]:
        """

        :param dataset_directory:
        :param ds:
        :return: annotations: List[List[Dict]]
        """
        # Use rasterio to compute geometric transformation parameters
        ulx, xres, _, uly, _, yres = ds.get_transform()

        # Size of Ortho in geometric scale
        img_h, img_w = ds.height, ds.width
        orthox = xres * img_w
        orthoy = yres * img_h

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
        bad_segments = 0
        print(f'Tiling Shapes for {os.path.basename(dataset_directory)}')
        for shape_rec in tqdm(shape_recs):
            shp = shape_rec.shape
            if shp.shapeType == 5:  # 5 - polygon

                class_name = shape_rec.record.segClass
                if class_name[-1] == '2':
                    class_name = class_name[:-1]
                # Update class dict
                if class_name not in self.classes:
                    self.classes[class_name] = len(self.classes)

                # transform polygon and bounding into image coordinate system
                rescaled_poly = [[rescale_x(x), rescale_y(y)] for x, y in fix_polygon_tail(shp.points)]
                if len(rescaled_poly) < 3:
                    bad_segments += 1
                    continue

                shape_bbox = [rescale_x(shp.bbox[0]), rescale_y(shp.bbox[3]),
                              rescale_x(shp.bbox[2]), rescale_y(shp.bbox[1])]

                if not bbox_filtering_function(shape_bbox):
                    bad_segments += 1
                    continue

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
        print(f'{bad_segments} of {len(shape_recs)} segments filtered')
        return annotations

    def tile_dataset(self, tile_filtering_function=lambda tile: True,
                     annotation_filtering_function=lambda anon: True,
                     bbox_filtering_function=lambda bbox: True, train_limit: int = 80, no_train: bool = False):
        """

        :param annotation_filtering_function:
        :param no_train:
        :param bbox_filtering_function:
        :param tile_filtering_function:
        :param train_limit:
        :raises FileNotFoundError: if a single file matching *ortho.tif cannot be found in the dataset path
        """
        if no_train:
            train_limit = 100
        self.build_output_dir_structure(no_train)
        for dataset_directory, dataset_name in zip(self.dataset_input_paths, self.dataset_names):
            ortho_name = glob(os.path.join(dataset_directory, '*ortho.tif'))
            if len(ortho_name) != 1:
                raise FileNotFoundError(f'Either 0 or > 1 files matching *ortho.tif found in {dataset_directory}')
            ortho_name = ortho_name[0]

            chm_name = glob(os.path.join(dataset_directory, '*CHM.tif'))
            if len(chm_name) != 1:
                raise FileNotFoundError(f'Either 0 or > 1 files matching *CHM.tif found in {dataset_directory}')
            chm_name = chm_name[0]

            print(f'Reading Ortho: {ortho_name}')
            ortho = imread(ortho_name)

            ds = rasterio.open(ortho_name, 'r')
            # Use rasterio to compute geometric transformation parameters
            x_geo_ref, xres, _, y_geo_ref, _, yres = ds.get_transform()

            if self.compute_means:
                mean = np.mean(ortho[:, :, :-1].reshape(-1, 3), 0)
                print(f'Pixel Mean is {mean}')
                self.ortho_means.append(mean)

            print(f'Reading CHM: {chm_name}')
            chm = imread(chm_name)

            print('Cleaning NaN values from CHM')
            chm = np.where(chm == -np.inf, 0, chm)
            print('Rescaling to [0, 255]')
            chm = (255 * (chm - np.min(chm)) / np.max(chm)).astype(ortho.dtype)

            if self.compute_means:
                mean = np.mean(chm)
                print(f'Pixel Mean is {mean}')
                self.chm_means.append(mean)

            assert (chm.shape[:2] == ortho.shape[:2]), 'Ortho and CHM are different shapes'
            assert ortho.shape[-1] == 4, 'Ortho should contain a 4th channel'

            if not no_train:
                annotations = self._tile_segments(dataset_directory=dataset_directory,
                                                  bbox_filtering_function=bbox_filtering_function, ds=ds)

            img_h, img_w = ortho.shape[:2]
            tile_number, train_id, test_id, tile_id = 0, 0, 0, 0
            train_record, test_record, tile_record = [], [], []
            offsets = {}
            x_tiles, y_tiles = ceil(img_w / self.dx), ceil(img_h / self.dy)
            offsets['x_tiles'], offsets['y_tiles'] = x_tiles, y_tiles

            num_tiles = x_tiles * y_tiles
            bad_tiles = 0
            for y in range(0, img_h, self.dy):
                for x in range(0, img_w, self.dx):
                    x_c = x + self.tile_width if x + self.tile_width <= img_w else img_w
                    y_c = y + self.tile_height if y + self.tile_height <= img_h else img_h

                    tile = ortho[y:y_c, x:x_c]
                    tile[:, :, -1] = chm[y:y_c, x:x_c]
                    make_dir = lambda name, id_: os.path.join(self.output_dir, name, dataset_name,
                                                              f'tile_{id_}.png')

                    if no_train:
                        image_output_dir = make_dir('tiles', tile_id)
                        tile_id += 1
                        offsets[image_output_dir] = (x_geo_ref + x * xres, y_geo_ref + y * yres)
                        cv2.imwrite(image_output_dir, tile)

                    elif tile_filtering_function(tile) and annotation_filtering_function(annotations[tile_number]):
                        annotation = lambda image_output_dir: {'file_name': image_output_dir,
                                                               'image_id': tile_number,
                                                               'width': (x_c - x),
                                                               'height': (y_c - y),
                                                               'annotations': annotations[tile_number]}
                        if np.random.randint(0, 100) <= train_limit:
                            image_output_dir = make_dir('train', train_id)
                            train_id += 1
                            train_record.append(annotation(image_output_dir))
                        else:
                            image_output_dir = make_dir('test', test_id)
                            test_id += 1
                            test_record.append(annotation(image_output_dir))
                        cv2.imwrite(image_output_dir, tile)

                    else:
                        bad_tiles += 1

                    tile_number += 1
                    if tile_number % 200 == 0:
                        print(f'Tile # {tile_number} of {num_tiles} created for {dataset_name}')

            print(f'{dataset_name} complete. {bad_tiles} of {num_tiles} tiles removed.')

            offsets['transform'] = (xres, yres)

            if no_train:
                with open(os.path.join(self.output_dir, 'tiles', dataset_name, 'offsets.json'), 'w') as f:
                    f.write(json.dumps(offsets))
            else:
                for root, rec in zip(['train', 'test'], [train_record, test_record]):
                    with open(os.path.join(self.output_dir, root, dataset_name, 'segs.json'), 'w') as f:
                        f.write(json.dumps(rec))

                print(f'Writing Classes {self.classes}')
                with open(os.path.join(self.output_dir, 'classes.json'), 'w') as f:
                    f.write(json.dumps(self.classes))

        if self.compute_means:
            print(f'Means {{CHM: {np.mean(self.chm_means)}, Ortho: f{np.mean(self.ortho_means, axis=0)}}}')

    def build_output_dir_structure(self, no_train: bool = False):
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
                f'Please clear it manually, with DataTiler.cleanup() or specify a different directory.')
            raise FileExistsError
        if no_train:
            sub_dirs = ['tiles']
        else:
            sub_dirs = ['train', 'test']
        for dir_name in sub_dirs:
            os.mkdir(os.path.join(self.output_dir, dir_name))
            for dataset_name in self.dataset_names:
                os.mkdir(os.path.join(self.output_dir, dir_name, dataset_name))

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


def remove_no_annotations(an):
    if len(an) > 0:
        return True
    return False


def remove_small_segment_coverage(thresh: float = .5):
    def f(an):
        return (lambda x: ((np.max(x[2]) - np.min(x[0])) * (np.max(x[3]) - np.min(x[1]))) / (640 ** 2))(
            np.array([x['bbox'] for x in an]).transpose()) > thresh

    return f


def if_all(funcs):
    """
    Short circuit evaluation to create a function that is the and of many functions
    :param funcs:
    :return: bool
    """

    def f(tile, annotation):
        for func in funcs:
            if not func(tile, annotation):
                return False
        return True

    return f


def remove_small_bboxes(thresh: 1000):
    """

    :param thresh: Minimum bounding box area to keep
    :return:
    """

    def f(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) > thresh

    return f


def non_zero_ratio(x):
    return np.count_nonzero(x) / len(x)


def ignore_black_tiles(thresh: float = .99):
    def f(tile):
        return non_zero_ratio(tile[:, :, :-1].flatten()) > thresh

    return f


if __name__ == '__main__':
    dt = DataTiler('/home/ubuntu/datasets', '/home/ubuntu/RGBD-Training-Data', cleanup_on_init=True,
                   tile_width=640,
                   tile_height=640, horizontal_overlay=320, vertical_overlay=320, dataset_regex='*')

    dt.tile_dataset(
        tile_filtering_function=ignore_black_tiles(thresh=.99),
        annotation_filtering_function=lambda an: remove_no_annotations(an) and remove_small_segment_coverage()(an),
        bbox_filtering_function=remove_small_bboxes(1000),
        no_train=False)
