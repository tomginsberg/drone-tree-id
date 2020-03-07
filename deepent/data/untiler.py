from glob import glob

import shapefile
from shapely.geometry import Polygon
from tqdm import tqdm

from detectron2.utils.visualizer import GenericMask


class PolygonRecord:
    def __init__(self, num_tiles, x_tiles):
        self.polygons = [[] for _ in range(num_tiles)]
        self.meta = [[] for _ in range(num_tiles)]
        self.x_tiles = x_tiles

    def put(self, tile_num, polygon, id_, area, cls):
        self.polygons[tile_num].append(polygon)
        self.meta[tile_num].append((id_, area, cls))

    def get_neighbours(self, tile_num):
        if tile_num == 0:
            # First tile, ie no neighbours
            return []
        if tile_num < self.x_tiles:
            # First row, neighbours are only the tiles one left
            return self.polygons[tile_num - 1]
        if tile_num > self.x_tiles:
            # Past first row
            if tile_num % self.x_tiles == 0:
                # First col only neighbour is above
                return self.polygons[tile_num - self.x_tiles]
            else:
                # Non trivial case, need to check all three neighbours
                return self.polygons[tile_num - 1] + self.polygons[tile_num - self.x_tiles] + self.polygons[
                    tile_num - self.x_tiles - 1]

    def poly_meta(self):
        return zip(self.polygons, self.meta)


class Untiler:
    def __init__(self, predictor):
        self._predictor = predictor

    def __call__(self, path_to_tiles: str, output: str, epsg_ref="32610", *args, **kwargs):
        tree_id = 0

        with open(os.path.join(path_to_tiles, 'offsets.json'), 'r') as f:
            offsets = json.loads(f.read())
        x_scale, y_scale = offsets['transform']

        tiles = glob(os.path.join(path_to_tiles, "*.png"))
        poly_record = PolygonRecord(num_tiles=len(tiles), x_tiles=offsets['x_tiles'])
        removed_polys, total_polys = 0, 0

        for tile_num, tile in tqdm(enumerate(tiles)):
            img = cv2.imread(tile)
            width, height = img.shape[1], img.shape[0]
            x_shift, y_shift = offsets[os.path.realpath(tile)]
            predictions = self._predictor(img)
            predictions = predictions["instances"].to("cpu")
            if predictions.has("pred_masks"):
                for (polygon, area, cls) in format_predictions(predictions, height, width):
                    total_polys += 1
                    if len(polygon) < 4:
                        continue
                    neighbours = poly_record.get_neighbours(tile_num)
                    next_poly = Polygon(affine_polygon(polygon, x_scale, y_scale, x_shift, y_shift))
                    if new_polygon_q(next_poly, neighbours, iou_thresh=.90, area_thresh=3):
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
                    shp.poly(list(poly.exterior.coords))
                    shp.record(tree_id, area, cls)

        with open(f'{output}.prj', "w+") as prj:
            epsg = getWKT_PRJ(epsg_ref)
            prj.write(epsg)


def new_polygon_q(poly, neighbours, iou_thresh: .85, area_thresh=3):
    if poly.area < area_thresh:
        return False
    for neighbour in neighbours:
        if neighbour.intesection(poly).area / neighbour.union(poly) > iou_thresh:
            return False
        if neighbour.contains(poly):
            return False
        if neighbour.within(poly):
            return False
    return True


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

    predictor = DefaultPredictor(cfg)

    ut = Untiler(predictor)

    ut(path_to_tiles='/home/ubuntu/CPTA-nInferenceTiles/tiles/CPT2a-n',
       output='/home/ubuntu/drone-tree-id/output/shapefiles/rgbnonduplicate/cpta-n')