from glob import glob

import shapefile
from shapely.geometry import Polygon
from shapely.errors import TopologicalError
from tqdm import tqdm

from detectron2.utils.visualizer import GenericMask
import IPython
from shutil import copyfile


class PolygonRecord:
    def __init__(self, num_tiles, x_tiles):
        self.polygons = [[] for _ in range(num_tiles)]
        self.meta = [[] for _ in range(num_tiles)]
        self.x_tiles = x_tiles

    def put(self, tile_num, polygon, id_, area, cls):
        self.polygons[tile_num].append(polygon)
        self.meta[tile_num].append((id_, area, cls))

    def get_neighbours(self, tile_num, lookahead=False):
        xw = self.x_tiles
        # left, top, top left,
        # top right, bottom, right, bottom right, bottom left
        behind, infront = [tile_num - 1, tile_num - xw, tile_num - xw - 1], [tile_num - xw + 1, tile_num + xw,
                                                                             tile_num + 1,
                                                                             tile_num + 1 + xw, tile_num - 1 + xw]
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

    # if tile_num == 0:
    #     # First tile, ie no neighbours
    #     if lookahead
    #         return []
    # if tile_num < self.x_tiles:
    #     # First row, neighbours are only the tiles one left
    #     if lookahead:
    #         return self.polygons[tile_num - 1] + self.polygons[tile_num + 1] + self.polygons[
    #             tile_num + self.x_tiles - 1] + self.polygons[tile_num + self.x_tiles + 1]
    #     return self.polygons[tile_num - 1]
    # if tile_num > self.x_tiles:
    #     # Past first row
    #     if tile_num % self.x_tiles == 0:
    #         # First col only neighbour is above
    #         return self.polygons[tile_num - self.x_tiles]
    #     else:
    #         # Non trivial case, need to check all three neighbours
    #         return self.polygons[tile_num - 1] + self.polygons[tile_num - self.x_tiles] + self.polygons[
    #             tile_num - self.x_tiles - 1]

    def poly_meta(self):
        return zip(self.polygons, self.meta)

    def _get_many_polygons(self, list_of_indices):
        polys = []
        for idx in list_of_indices:
            if 0 <= idx < len(self.polygons):
                polys += self.polygons[idx]
        return polys


class Untiler:
    def __init__(self, primary_predictor, secondary_predictor=None):
        self.primary_predictor = primary_predictor
        self.secondary_predictor = secondary_predictor

    def __call__(self, path_to_tiles: str, output: str, epsg_ref="32610", *args, **kwargs):
        tree_id = 0

        with open(os.path.join(path_to_tiles, 'offsets.json'), 'r') as f:
            offsets = json.loads(f.read())
        x_scale, y_scale = offsets['transform']

        tiles = sorted(glob(os.path.join(path_to_tiles, "*.png")),
                       key=lambda x: int(os.path.basename(x).split('_')[-1][:-4]))
        poly_record = PolygonRecord(num_tiles=len(tiles), x_tiles=offsets['x_tiles'])
        removed_polys, total_polys = 0, 0

        for tile_num, tile in tqdm(enumerate(tiles)):
            img = cv2.imread(tile, cv2.IMREAD_UNCHANGED)
            width, height = img.shape[1], img.shape[0]
            x_shift, y_shift = offsets[os.path.realpath(tile)]
            predictions = self.primary_predictor(img)
            predictions = predictions["instances"].to("cpu")
            neighbours = poly_record.get_neighbours(tile_num, lookahead=False)

            if predictions.has("pred_masks"):
                for (polygon, area, cls) in format_predictions(predictions, height, width):
                    total_polys += 1
                    if len(polygon) > 4:
                        next_poly = Polygon(affine_polygon(polygon, x_scale, y_scale, x_shift, y_shift)).simplify(0.1)
                        if new_polygon_q(next_poly, neighbours, iou_thresh=.70, area_thresh=3):
                            poly_record.put(tile_num, next_poly, tree_id,
                                            area * x_scale * y_scale, cls)
                            tree_id += 1
                        else:
                            removed_polys += 1

        if self.secondary_predictor is not None:
            for tile_num, tile in tqdm(enumerate(tiles)):
                img = cv2.imread(tile)
                width, height = img.shape[1], img.shape[0]
                x_shift, y_shift = offsets[os.path.realpath(tile)]
                predictions = self.secondary_predictor(img)
                predictions = predictions["instances"].to("cpu")
                neighbours = poly_record.get_neighbours(tile_num, lookahead=True)

                if predictions.has("pred_masks"):
                    for (polygon, area, cls) in format_predictions(predictions, height, width):
                        total_polys += 1
                        if len(polygon) > 4:
                            next_poly = Polygon(affine_polygon(polygon, x_scale, y_scale, x_shift, y_shift)).simplify(
                                0.1)
                            if new_polygon_q(next_poly, neighbours, iou_thresh=.70):
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
        # with open(f'{output}.prj', "w+") as prj:
        #     epsg = getWKT_PRJ(epsg_ref)
        #     prj.write(epsg)


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


def getWKT_PRJ(epsg_code):
    import urllib, ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    wkt = urllib.request.urlopen("http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(epsg_code))
    remove_spaces = str(wkt.read()).replace(" ", "")
    output = remove_spaces.replace("\n", "")
    return output


if __name__ == '__main__':
    import json
    import cv2
    import os
    import numpy as np

    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor, default_setup

    from deepent.config import add_deepent_config

    config_file = 'configs/deepent_rcnn_R_50_FPN.yaml'
    threshold = 0.5
    model = 'output/baseline_25_01_2020/model_0054999.pth'
    samples = 1
    type_ = 'many'
    opts = []


    class Args:
        def __init__(self, conf, tr, mod, sam, ty, opt):
            self.config_file = conf
            self.threshold = tr
            self.model = mod
            self.samples = sam
            self.type = ty
            self.opts = []
            self.output = None


    args = Args(config_file, threshold, model, samples, type_, opts)

    cfg = get_cfg()
    add_deepent_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.freeze()
    default_setup(cfg, args)

    secondary_predictor = DefaultPredictor(cfg)

    config_file = 'configs/deepent_fuse_rcnn_R_50_FPN.yaml'
    threshold = 0.5
    model = 'output/baseline_fuse_07_02_2020/model_0089999.pth'
    samples = 1
    type_ = 'many'
    opts = []


    class Args:
        def __init__(self, conf, tr, mod, sam, ty, opt):
            self.config_file = conf
            self.threshold = tr
            self.model = mod
            self.samples = sam
            self.type = ty
            self.opts = []
            self.output = None


    args = Args(config_file, threshold, model, samples, type_, opts)

    cfg = get_cfg()
    add_deepent_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.freeze()
    default_setup(cfg, args)
    from tools.predictor import RGBDPredictor

    primary_predictor = RGBDPredictor(cfg)

    ut = Untiler(primary_predictor, secondary_predictor)

    ut(path_to_tiles='/home/ubuntu/twister_inference/tiles/twister',
       output='/home/ubuntu/drone-tree-id/output/shapefiles/rgbnonduplicate/twister')
