#!/usr/bin/env python3

import os
import argparse
from typing import List, Union, Dict, Tuple

from detectron2.structures import BoxMode
import itertools
import cv2
import shapefile
import rasterio
import numpy as np
from data_processing.tile_dataset import TiledDataset


def shapefile_to_coco_dict(dataset_path: str) -> Tuple[List[str], List[Dict]]:
    """
    Convert Shapefile dataset to COCO format.
    See https://detectron2.readthedocs.io/tutorials/datasets.html for using custom dataset with detectron2.

    :param
    dataset_path:
        str: root directory of dataset containing image (*ortho-resample.tif) and corresponding shapefile
        (in Segments directory) example: //Google Drive/UBC_EngCapstone/sample_data/CPT2a-n, here the dataset is CPT2a-n
    :return:
    classes:
        list, str: list of segment classes
    dataset_dicts:
        dict: COCO format dict of dataset annotations/segments
    """
    dataset_name = os.path.basename(dataset_path)
    record = {}

    # Ortho and Shapefile geospacial transformation info
    shpf = shapefile.Reader(os.path.join(dataset_path, "Segments/" + dataset_name + "_dom-poly"))
    shape_recs = shpf.shapeRecords()
    # lower left (x,y), upper right (x,y)
    bbox = shpf.bbox

    ds = rasterio.open(os.path.join(dataset_path, dataset_name + '_ortho-resample.tif'), 'r')
    # upper left (x,y), resolution, skew _
    ulx, xres, _, uly, _, yres = ds.get_transform()
    # Size of Ortho in geometric scale
    orthox = xres * ds.width
    orthoy = yres * ds.height

    filename = os.path.join(dataset_path, dataset_name + "_ortho-resample.tif")
    height, width = cv2.imread(filename).shape[:2]

    # don't wanna fuck with the flow, but consider making this an object
    record["file_name"] = filename
    record["image_id"] = 0
    record["height"] = height
    record["width"] = width

    # COCO wants annotation points absolute from top left corner scaled to image pixels, see bbox_mode
    # https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.BoxMode.
    # Need to offset data from [(0,0) -> abs(width, height)]
    # Global offsets
    minx = bbox[0]
    maxy = bbox[3]
    # Offset of Shapefile from top left corner of Ortho [0, width/height] in geometric scale
    offsetx = minx - ulx
    offsety = maxy - uly
    # Scale is ratio of image pixel width/height to geometric width/height in raster
    scalex = width / orthox
    scaley = height / orthoy
    rescale_x = lambda x: (x - minx + offsetx) * scalex
    rescale_y = lambda y: (y - maxy + offsety) * scaley

    objs = []
    classes = ['tree']
    for shape_rec in shape_recs:
        shp = shape_rec.shape
        if shp.shapeType == 5:  # 5 - polygon
            # poly = [(rescale_x(x), rescale_y(y)) for (x, y) in shp.points]
            poly = fix_polygon_tail(shp.points)
            poly = np.array([[rescale_x(x), rescale_y(y)] for (x, y) in poly])
            # poly = list(itertools.chain.from_iterable(poly))

            # if rec.segClass not in classes:
            #     classes.append(rec.segClass)
            objs.append({
                # scaley < 0, offsety < 0
                "bbox": [rescale_x(shp.bbox[0]), rescale_y(shp.bbox[1]),
                         rescale_x(shp.bbox[2]), rescale_y(shp.bbox[3])],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": poly,
                "category_id": 0,
                # "category_id": rec.segClass, # classification enabled, we can force each segment to a
                # single tree class if needed
                "iscrowd": 0
                # iscrowd groups individual objects of the same kind into a single segment. For tree segmentation,
                # we want to isolate trees
            })
    record["annotations"] = objs
    td = TiledDataset(record, width=600, height=600, w_overlay=300, h_overlay=300)

    dataset_dicts = td.tile_polygons()
    return classes, dataset_dicts


def main():
    from detectron2.data import DatasetCatalog, MetadataCatalog
    parser = argparse.ArgumentParser(description='Register Custom Shapefile dataset for Detectron2.')
    parser.add_argument('dataset_path', type=str, help='Path to root directory for all datasets.')
    args = parser.parse_args()
    dataset_path = 'datasets/CPT2a-n'
    datasets = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and os.path.isdir(
        os.path.join(dataset_path, d + '/Segments'))]
    for dataset in datasets:
        if dataset not in DatasetCatalog.list():
            classes, dataset_dicts = shapefile_to_coco_dict(os.path.join(dataset_path, dataset))
            DatasetCatalog.register(dataset, lambda: dataset_dicts)
            MetadataCatalog.get(dataset).set(thing_classes=classes)


def fix_polygon_tail(polygon):
    first = polygon[0]
    new_poly = []
    for i, p in enumerate(polygon):
        new_poly.append(p)
        if i > 0 and p == first:
            break
    return np.array(new_poly)


if __name__ == "__main__":
    main()
