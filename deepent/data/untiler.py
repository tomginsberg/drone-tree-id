import json
import os
from glob import glob
from shutil import copyfile

import cv2
import numpy as np
import shapefile
from shapely.errors import TopologicalError
from shapely.geometry import Polygon
from tqdm import tqdm

from detectron2.utils.visualizer import GenericMask
from tools.predictor import RGBDPredictor
import IPython

class PolygonRecord:
    def __init__(self, num_tiles, x_tiles):
        self.polygons = [[] for _ in range(num_tiles)]
        self.meta = [[] for _ in range(num_tiles)]
        self.x_tiles = x_tiles

    def put(self, tile_num, polygon, id_, area, cls):
        self.polygons[tile_num].append(polygon)
        self.meta[tile_num].append([id_, area, cls])

    def get_neighbours(self, tile_num, lookahead=False):
        xw = self.x_tiles
        # left, top, top left,
        # top right, bottom, right, bottom right, bottom left
        behind = [tile_num - 1, tile_num - xw, tile_num - xw - 1]
        infront = [tile_num - xw + 1, tile_num + xw, tile_num + 1, tile_num + 1 + xw, tile_num - 1 + xw]

        if lookahead:
            if tile_num % xw == 0:
                # left border, take above and all in front but bottom left
                indices = [behind[1]] + infront[:-1]
            elif tile_num + 1 % xw == 0:
                # right border, take all behind and bottom + bottom left
                indices = behind + [infront[1], infront[-1]]
            else:
                indices = behind + infront
            indices += [tile_num]
        else:
            # No lookahead
            # If on the left border only take above
            if tile_num % xw == 0:
                indices = [behind[1]]
            else:
                indices = behind

        return self._get_many_polygons(indices)

    def poly_meta(self):
        return zip(self.polygons, self.meta)

    def _get_many_polygons(self, list_of_indices):
        polys = []
        for idx in list_of_indices:
            if 0 <= idx < len(self.polygons):
                polys += self.polygons[idx]
        return polys


class Untiler:
    def __init__(self, predictors):
        self.predictors = predictors

    def predict_and_untile(self, path_to_tiles: str, output: str):
        tree_id = 0

        with open(os.path.join(path_to_tiles, 'offsets.json'), 'r') as f:
            offsets = json.loads(f.read())
        x_scale, y_scale = offsets['transform']

        tiles = sorted(glob(os.path.join(path_to_tiles, "*.png")),
                       key=lambda x: int(os.path.basename(x).split('_')[-1][:-4]))
        poly_record = PolygonRecord(num_tiles=len(tiles), x_tiles=offsets['x_tiles'])
        removed_polys, total_polys = 0, 0

        for i, predictor in enumerate(self.predictors):
            for tile_num, tile in tqdm(enumerate(tiles)):
                if isinstance(predictor, RGBDPredictor):
                    img = cv2.imread(tile, cv2.IMREAD_UNCHANGED)
                else:
                    img = cv2.imread(tile)
                width, height = img.shape[1], img.shape[0]
                x_shift, y_shift = offsets[os.path.realpath(tile)]
                predictions = predictor(img)
                predictions = predictions["instances"].to("cpu")
                neighbours = poly_record.get_neighbours(tile_num, lookahead=(i > 0))

                if predictions.has("pred_masks"):
                    for (polygon, area, cls) in format_predictions(predictions, height, width):
                        total_polys += 1
                        if len(polygon) > 4:
                            next_poly = Polygon(affine_polygon(polygon, x_scale, y_scale, x_shift, y_shift)).simplify(
                                0.2)
                            if new_polygon_q(next_poly, neighbours, iou_thresh=.70, area_thresh=3):
                                poly_record.put(tile_num, next_poly, tree_id,
                                                area * x_scale * y_scale, cls)
                                tree_id += 1
                            else:
                                removed_polys += 1

        print(f'Inference done. {removed_polys}/{total_polys} polygons removed')
        with shapefile.Writer(output) as shp:
            shp.shapeType = 5  # set shape type to polygons
            shp.field('treeID', 'N', 24, 15)
            shp.field('polyArea', 'N', 24, 15)
            shp.field('segClass', 'C', 80, 0)
            for polys, metas in tqdm(poly_record.poly_meta()):
                for poly, (tree_id, area, cls) in zip(polys, metas):
                    shp.poly([list(poly.exterior.coords)])
                    shp.record(tree_id, area, cls)

        copyfile('deepent/data/resources/generic.prj', f'{output}.prj')


def new_polygon_q(poly, neighbours, iou_thresh: .85, area_thresh=2):
    if poly.area < area_thresh:
        return False
    for neighbour in neighbours:
        try:
            intersection = neighbour.intersection(poly).area
            if intersection / neighbour.union(poly).area > iou_thresh:
                return False
            if intersection / neighbour.area > iou_thresh:
                return False
            if intersection / poly.area > iou_thresh:
                return False
        except TopologicalError:
            return False
    return True


def affine_polygon(polygon, x_scale, y_scale, x_shift, y_shift):
    """
    x and y scale -> should be inverse of scaling used to tile.
    shift in shapefile coordinates
    """
    x, y = polygon.transpose()
    return np.array([x * x_scale + x_shift, y * y_scale + y_shift]).transpose()


def format_predictions(predictions, height, width):
    masks = np.asarray(predictions.pred_masks)
    masks = [GenericMask(mask, height, width) for mask in masks]
    # polygon should not have holes (len(poly) = 1)
    polygons = [reshape_and_close_poly(mask.polygons[0]) for mask in masks]
    # boxes = predictions.pred_boxes if predictions.has("pred_boxes") else [mask.bbox() for mask in masks]
    classes = predictions.pred_classes if predictions.has("pred_classes") else [None for _ in masks]
    areas = [mask.area() for mask in masks]

    assert (len(polygons) == len(classes) == len(areas))

    return zip(polygons, areas, classes)


def reshape_and_close_poly(poly):
    poly = np.append(poly, poly[0:2])
    return np.reshape(poly, (len(poly) // 2, 2))
