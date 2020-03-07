import json
import os
from glob import glob

import cv2
import numpy as np
import shapefile

from detectron2.utils.visualizer import GenericMask


class Untiler:
    def __init__(self, predictor):
        self._predictor = predictor

    def __call__(self, path_to_tiles: str, output: str, epsg_ref="32610", *args, **kwargs):
        tree_id = 0

        with open(os.path.join(path_to_tiles, 'offsets.json'), 'r') as f:
            offsets = json.loads(f.read())

        tiles = glob(os.path.join(path_to_tiles, "*.png"))

        with shapefile.Writer(output) as shp:
            shp.shapeType = 5  # set shapetype to polygons
            shp.field('treeID', 'N', 24, 15)
            shp.field('polyArea', 'N', 24, 15)
            shp.field('segClass', 'C', 80, 0)

            x_scale, y_scale = offsets['transform']

            for tile in tiles:
                img = cv2.imread(tile)
                width, height = img.shape[1], img.shape[0]
                x_shift, y_shift = offsets[os.path.realpath(tile)]
                predictions = self._predictor(img)
                predictions = predictions["instances"].to("cpu")
                if predictions.has("pred_masks"):
                    for (polygon, area, cls) in format_predictions(predictions, height, width):
                        shp.poly(parts=affine_polygon(polygon, x_scale, y_scale, x_shift, y_shift))
                        # TODO: convert index class to string
                        shp.record(tree_id, area * x_scale * y_scale, cls)
                        tree_id += 1

        with open(f'{output}.prj', "w+") as prj:
            epsg = getWKT_PRJ(epsg_ref)
            prj.write(epsg)


def affine_polygon(polygon, x_scale, y_scale, x_shift, y_shift):
    """
    x and y scale -> should be inverse of scaling used to tile.
    shift in shapefile coordinates
    """
    x, y = polygon.transpose()
    return [np.array([x * x_scale + x_shift, y * y_scale + y_shift]).transpose()]


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


def getWKT_PRJ(epsg_code):
    import urllib, ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    wkt = urllib.request.urlopen("http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(epsg_code))
    remove_spaces = str(wkt.read()).replace(" ", "")
    output = remove_spaces.replace("\n", "")
    return output
