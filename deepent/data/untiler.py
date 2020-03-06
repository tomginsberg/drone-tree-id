import os
from glob import glob

import cv2
import numpy as np
import shapefile

from detectron2.utils.visualizer import GenericMask


class Untiler:
    def __init__(self, predictor, path_to_tiles: str, output: str):
        self._predictor = predictor
        self.output = output
        self._tree_id = 0
        self._path_to_tiles = path_to_tiles

    def __call__(self, *args, **kwargs):

        tiles = glob(os.path.join(self._path_to_tiles, "*.png"))

        with shapefile.Writer(self.output) as shp:
            shp.shapeType = 5  # set shapetype to polygons
            shp.field('treeID', 'N', 24, 15)
            shp.field('polyArea', 'N', 24, 15)
            shp.field('segClass', 'C', 80, 0)
        x_scale, y_scale = load_scales()
        for tile in tiles:
            img = cv2.imread(tile)
            width, height = img.shape[1], img.shape[0]
            # TODO: implement
            x_shift, y_shift = load_transform(tile)
            predictions = self._predictor(img)
            predictions = predictions["instances"].to("cpu")
            if predictions.has("pred_masks"):
                for (polygon, area, cls) in format_predictions(predictions, height, width):
                    shp.poly(affine_polygon(polygon, x_scale, y_scale, x_shift, y_shift))
                    # TODO: is class the id number or the string????????
                    shp.record(self._tree_id, area * x_scale * y_scale, cls)
                    self._tree_id += 1


def affine_polygon(polygon, x_scale, y_scale, x_shift, y_shift):
    """
    x and y scale -> should be inverse of scaling used to tile.
    shift in shapefile coordinates
    """
    rescale_x = lambda x: (x * x_scale + x_shift)
    rescale_y = lambda y: (y * y_scale + y_shift)
    # TODO: deal with nested polygons(i.e. holes)?
    rescaled_poly = [[rescale_x(x), rescale_y(y)] for x, y in polygon]
    return rescaled_poly


def format_predictions(predictions, height, width):
    masks = np.asarray(predictions.pred_masks)
    masks = [GenericMask(mask, height, width) for mask in masks]

    polygons = [mask.mask_to_polygons(mask) for mask in masks]
    # boxes = predictions.pred_boxes if predictions.has("pred_boxes") else [mask.bbox() for mask in masks]
    classes = predictions.pred_classes if predictions.has("pred_classes") else [None for _ in masks]
    areas = [mask.sum() for mask in masks]

    assert (len(polygons) == len(classes) == len(areas))

    return zip(polygons, areas, classes)

