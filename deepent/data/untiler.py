import json
import os
from glob import glob

import cv2
import numpy as np
import shapefile

from detectron2.utils.visualizer import GenericMask
from tqdm import tqdm

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

            for tile in tqdm(tiles):
                img = cv2.imread(tile, cv2.IMREAD_UNCHANGED)
                width, height = img.shape[1], img.shape[0]
                x_shift, y_shift = offsets[os.path.realpath(tile)]
                predictions = self._predictor(img)
                predictions = predictions["instances"].to("cpu")
                if predictions.has("pred_masks"):
                    for (polygon, area, cls) in format_predictions(predictions, height, width):
                        shp.poly(affine_polygon(polygon, x_scale, y_scale, x_shift, y_shift))
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

if __name__ == '__main__':
    import json
    import matplotlib.pyplot as plt
    import cv2
    import os
    import numpy as np
    import random

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog, DatasetCatalog

    from deepent.data.register_datasets import register_datasets
    from deepent.config import add_deepent_config

    config_file = 'configs/deepent_fuse_rcnn_R_50_FPN.yaml'
    threshold = 0.5
    model = 'output/baseline_fuse_07_02_2020/model_0049999.pth'
    samples = 1
    dataset = 'CPT2a-n_train'
    type_ = 'many'
    opts = []


    class Args:
        def __init__(self, conf, tr, mod, sam, ty, opt, dts):
            self.config_file = conf
            self.threshold = tr
            self.model = mod
            self.samples = sam
            self.dataset = dts
            self.type = ty
            self.opts = []
            self.output = None


    args = Args(config_file, threshold, model, samples, type_, opts, dataset)

    cfg = get_cfg()
    add_deepent_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.freeze()
    default_setup(cfg, args)

    from tools.predictor import RGBDPredictor

    predictor = RGBDPredictor(cfg)

    ut = Untiler(predictor)

    ut(path_to_tiles='/home/ubuntu/RGBD-Tree-Segs-Clean/tiles/CPT2a-n',
       output='/home/ubuntu/drone-tree-id/output/shapefiles/rgbd/rgbdcpta-n')
