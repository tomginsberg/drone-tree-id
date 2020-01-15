#!/usr/bin/env python3

import os
import argparse
from typing import List, Union, Dict, Tuple
import json
from detectron2.structures import BoxMode
import itertools
import cv2
from skimage.io import imread
import shapefile
import numpy as np
from data_processing.tile_dataset import TiledDataset
import glob
from tqdm import tqdm
import rasterio

img_id = 0


def shapefile_to_coco_dict(dataset_path: str) -> Tuple[List[str], List[Dict]]:
    global img_id
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
    # shpf = shapefile.Reader(os.path.join(dataset_path, "Segments/" + dataset_name + "_dom-poly"))
    shpf = shapefile.Reader(os.path.join(dataset_path, "simple/segs_simple"))

    shape_recs = shpf.shapeRecords()
    # lower left (x,y), upper right (x,y)
    bbox = shpf.bbox

    # ds = rasterio.open(os.path.join(dataset_path, dataset_name + '_ortho-resample.tif'), 'r')
    print(f'Reading Raster for {dataset_name}')
    ds = rasterio.open(os.path.join(dataset_path, dataset_name + '_ortho.tif'), 'r')

    # upper left (x,y), resolution, skew _
    ulx, xres, _, uly, _, yres = ds.get_transform()
    # Size of Ortho in geometric scale
    orthox = xres * ds.RasterXSize
    orthoy = yres * ds.RasterYSize

    filename = os.path.join(dataset_path, dataset_name + "_ortho.tif")
    height, width = imread(filename).shape[:2]

    record["file_name"] = filename
    record["image_id"] = img_id
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
    print(f'Rescaling Segments for {dataset_name}')
    for shape_rec in tqdm(shape_recs):
        shp = shape_rec.shape
        if shp.shapeType == 5:  # 5 - polygon
            poly = fix_polygon_tail(shp.points)
            poly = np.array([[rescale_x(x), rescale_y(y)] for (x, y) in poly])

            # if rec.segClass not in classes:
            #     classes.append(rec.segClass)
            objs.append({
                # scaley < 0, offsety < 0
                "bbox": [rescale_x(shp.bbox[0]), rescale_y(shp.bbox[3]),
                         rescale_x(shp.bbox[2]), rescale_y(shp.bbox[1])],
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
    td.tile_ortho()

    dataset_dicts = td.tile_polygons()
    img_id += td.num_tiles

    return classes, dataset_dicts


def main():
    from detectron2.data import DatasetCatalog, MetadataCatalog
    # parser = argparse.ArgumentParser(description='Register Custom Shapefile dataset for Detectron2.')
    # parser.add_argument('dataset_path', type=str, help='Path to root directory for all datasets.')
    # args = parser.parse_args()

    dataset_path = 'FYBRData'

    for datasets in [glob.glob(f'{dataset_path}/train/*'), glob.glob(f'{dataset_path}/test/*')]:
        for dataset in datasets:
            with open(f'{dataset}/segs.json', 'r') as f:
                data = json.load(f)
            for tile in data:
                tile['bbox_mode'] = BoxMode.XYXY_ABS

            if dataset not in DatasetCatalog.list():
                print(f'{os.path.basename(dataset)}_{dataset.split("/")[-2]}')
                DatasetCatalog.register(f'{os.path.basename(dataset)}_{dataset.split("/")[-2]}', lambda: data)
                MetadataCatalog.get(dataset).set(thing_classes=['tree'])


def fix_polygon_tail(polygon):
    first = polygon[0]
    new_poly = []
    for i, p in enumerate(polygon):
        new_poly.append(p)
        if i > 0 and p == first:
            break
    return new_poly


def tile():
    datasets = glob.glob('datasets/*')
    for dataset in datasets:
        _, dataset_dict = shapefile_to_coco_dict(dataset)
        with open(f'tiled_{dataset}/segs.json', 'w') as f:
            f.write(json.dumps(dataset_dict))


if __name__ == "__main__":
    main()