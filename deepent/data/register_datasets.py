#!/usr/bin/env python3

import glob
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def register_datasets(dataset_path: str):
    def data_getter(data):
        def f():
            return data

        return f

    with open(os.path.join(dataset_path, 'classes.json'), 'r') as f:
        classes = json.load(f)
    classes = sorted(classes.keys(), key=lambda x: classes[x])

    for train_test in ['train', 'test']:
        datasets = glob.glob(os.path.join(dataset_path, train_test, '*'))
        for dataset in datasets:
            with open(os.path.join(dataset, 'segs.json'), 'r') as f:
                data = json.load(f)
            for tile in data:
                for annotation in tile['annotations']:
                    annotation['bbox_mode'] = BoxMode.XYXY_ABS

            if dataset not in DatasetCatalog.list():
                register_name = f'{os.path.basename(dataset)}_{train_test}'
                print(f'Registering {register_name}')
                DatasetCatalog.register(register_name, data_getter(data))
                MetadataCatalog.get(register_name).set(thing_classes=classes)


if __name__ == '__main__':
    try:
        register_datasets(dataset_path='/home/ubuntu/RGBD-Tree-Segs')
    except FileNotFoundError:
        register_datasets(dataset_path='RGBD-Tree-Segs')
