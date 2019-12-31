#!/usr/bin/env python3

import os
import argparse
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
import shapefile
from operator import itemgetter
import cv2

'''
Convert Shapefile dataset to COCO format.
See https://detectron2.readthedocs.io/tutorials/datasets.html for using custom dataset with detectron2.

path:
    str: root directory of dataset containing image (*ortho-resample.tif) and corresponding shapefile (in Segments directory)
    example: //Google Drive/UBC_EngCapstone/sample_data/CPT2a-n, here the dataset is CPT2a-n
    
TODO: Extend to multiple images and shapefiles in dataset, or clip single ortho here.

Returns:
    classes:
        list, str: list of segment classes of dataset
    dataset_dicts:
        dict: COCO format dict of dataset annotations/segments
'''
def get_dataset_dicts(path):
    dataset_name = os.path.basename(path)
    with shapefile.Reader(os.path.join(path, "Segments/"+ dataset_name + "_dom-poly")) as shp:
        shape_recs = shp.shapeRecords()
    dataset_dicts = []
    record = {}
    
    filename = os.path.join(path, dataset_name + "_ortho-resample.tif")
    height, width = cv2.imread(filename).shape[:2]
    
    record["file_name"] = filename
    record["image_id"] = 0
    record["height"] = height
    record["width"] = width

    # Normalize point values (remove offset) to be relative to (0,0).
    # bboxes = [x.shape.bbox for x in shape_recs]
    # minx = min(min(bboxes, key=itemgetter(0))[0], min(bboxes, key=itemgetter(2))[2])
    # miny = min(min(bboxes, key=itemgetter(1))[1], min(bboxes, key=itemgetter(3))[3])

    # Obtain global shift from GIS software
    # TODO: create lookup for each dataset to obtain these values
    minx = 568800
    miny = 5970600
    # Hard code ortho height and width for now.
    # The ortho and shapefile coordinates are scaled differently to resolution, need to figure this out.
    scaley = height/887.0
    scalex = width/737.5
      
    objs = []
    classes = []
    for shape_rec in shape_recs:
        rec = shape_rec.record
        shp = shape_rec.shape
        if shp.shapeTypeName is 'POLYGON':
            poly = [((x-minx)*scalex,(y-miny)*scaley) for (x,y) in shp.points]
            poly = list(itertools.chain.from_iterable(poly))

            if rec.segClass not in classes:
                classes.append(rec.segClass)

            obj = {
                "bbox": [(shp.bbox[0]-minx)*scalex, (shp.bbox[1]-miny)*scaley, (shp.bbox[2]-minx)*scalex, (shp.bbox[3]-miny)*scaley],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": rec.segClass, # classification enabled, we can force each segment to a single tree class if needed
                "iscrowd": 0 # iscrowd groups individual objects of the same kind into a single segment. For tree segmentation, we want to isolate trees
            }
            objs.append(obj)
    record["annotations"] = objs
    dataset_dicts.append(record)
    return classes, dataset_dicts

def main():
    from detectron2.data import DatasetCatalog, MetadataCatalog
    parser = argparse.ArgumentParser(description='Register Custom Shapefile dataset for Detectron2.')
    parser.add_argument('path', type=str, help='Path to root directory for all datasets.')
    args = parser.parse_args()
    datasets = [d for d in os.path.listdir(args.path) if isdir(join(args.path, d))]
    for dataset in datasets:
        classes, dataset_dicts = get_dataset_dicts(os.path.join(args.path, dataset))
        DatasetCatalog.register(dataset, lambda : dataset_dicts)
        MetadataCatalog.get(dataset).set(thing_classes=classes)

if __name__ == "__main__":
    main()
