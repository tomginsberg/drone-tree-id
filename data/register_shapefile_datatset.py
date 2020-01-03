#!/usr/bin/env python3

import os
import argparse
from detectron2.structures import BoxMode
import itertools
import cv2
import shapefile
import gdal


def shapefile_to_coco_dict(dataset_path: str) -> Tuple[List[str], List[Dict[str, Union[str, int, float]]]]:
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
    dataset_dicts = []
    record = {}

    # Ortho geometric info
    ds = gdal.Open(os.path.join(dataset_path, dataset_name + '_ortho-resample.tif'), gdal.GA_ReadOnly)
    # upper left (x,y), resolution, skew
    ulx, xres, _, uly, _, yres = ds.GetGeoTransform()
    # Size of Ortho in geometric scale
    orthox = xres * ds.RasterXSize
    orthoy = yres * ds.RasterYSize

    # Shapefile geometric info
    with shapefile.Reader(os.path.join(dataset_path, "Segments/" + dataset_name + "_dom-poly")) as shp:
        shape_recs = shp.shapeRecords()
        # lower left (x,y), upper right (x,y)
        bbox = shp.bbox

    filename = os.path.join(dataset_path, dataset_name + "_ortho-resample.tif")
    height, width = cv2.imread(filename).shape[:2]

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

    objs = []
    classes = ['tree']
    for shape_rec in shape_recs:
        rec = shape_rec.record
        shp = shape_rec.shape
        if shp.shapeTypeName is 'POLYGON':
            poly = [((x - minx + offsetx) * scalex, (y - maxy + offsety) * scaley) for (x, y) in shp.points]
            poly = list(itertools.chain.from_iterable(poly))

            # if rec.segClass not in classes:
            #     classes.append(rec.segClass)

            obj = {
                # scaley < 0, offsety < 0
                "bbox": [(shp.bbox[0] - minx + offsetx) * scalex, (shp.bbox[1] - maxy + offsety) * scaley,
                         (shp.bbox[2] - minx + offsetx) * scalex, (shp.bbox[3] - maxy + offsety) * scaley],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                # "category_id": rec.segClass, # classification enabled, we can force each segment to a
                # single tree class if needed
                "iscrowd": 0
                # iscrowd groups individual objects of the same kind into a single segment. For tree segmentation,
                # we want to isolate trees
            }
            objs.append(obj)
    record["annotations"] = objs
    dataset_dicts.append(record)
    return classes, dataset_dicts


def main():
    from detectron2.data import DatasetCatalog, MetadataCatalog
    parser = argparse.ArgumentParser(description='Register Custom Shapefile dataset for Detectron2.')
    parser.add_argument('dataset_path', type=str, help='Path to root directory for all datasets.')
    args = parser.parse_args()
    datasets = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) and os.path.isdir(
        os.path.join(dataset_path, d + '/Segments'))]
    for dataset in datasets:
        if dataset not in DatasetCatalog.list():
            classes, dataset_dicts = shapefile_to_coco_dict(os.path.join(dataset_path, dataset))
            DatasetCatalog.register(dataset, lambda: dataset_dicts)
            MetadataCatalog.get(dataset).set(thing_classes=classes)


if __name__ == "__main__":
    main()
