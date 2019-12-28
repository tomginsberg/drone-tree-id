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

def get_dataset_dicts(img_dir, classes):
    with shapefile.Reader(os.path.join(img_dir, "Segments/"+os.path.basename(img_dir) + "_dom-poly")) as shp:
        shape_recs = shp.shapeRecords()
    dataset_dicts = []
    record = {}
    
    filename = os.path.join(img_dir, os.path.basename(img_dir) + "_ortho-resample.tif")
    height, width = cv2.imread(filename).shape[:2]
    
    record["file_name"] = filename
    record["image_id"] = 0
    record["height"] = height
    record["width"] = width

    bboxes = [x.shape.bbox for x in shape_recs]
    minx = min(min(bboxes, key=itemgetter(0))[0], min(bboxes, key=itemgetter(2))[2])
    miny = min(min(bboxes, key=itemgetter(1))[1], min(bboxes, key=itemgetter(3))[3])
      
    objs = []
    for shape_rec in shape_recs:
        rec = shape_rec.record
        shp = shape_rec.shape
        if shp.shapeTypeName is 'POLYGON':
            poly = [(x-minx,y-miny) for (x,y) in shp.points]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [shp.bbox[0]-minx, shp.bbox[1]-miny, shp.bbox[2]-minx, shp.bbox[3]-miny],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(rec.segClass),
                "iscrowd": 0
            }
            objs.append(obj)
    record["annotations"] = objs
    dataset_dicts.append(record)
    return dataset_dicts

def main():
    from detectron2.data import DatasetCatalog, MetadataCatalog
    parser = argparse.ArgumentParser(description='Register Custom Shapefile dataset for Detectron2.')
    parser.add_argument('path', type=str, help='Path to dataset directory.')
    parser.add_argument('--classes', metavar='C', type=str, nargs='+', help='Segment classes')
    args = parser.parse_args()
    dataset = os.path.basename(args.path)
    if not args.classes:
        args.classes = ['tree']
    DatasetCatalog.register(dataset, lambda x=1: get_dataset_dicts(args.path, args.classes))
    MetadataCatalog.get(dataset).set(thing_classes=args.classes)

if __name__ == "__main__":
    main()
