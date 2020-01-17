import sys
sys.path.append('/home/ubuntu/drone-tree-id/')

import os
import random
import json
import matplotlib.pyplot as plt
import cv2

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

from deepent.data.register_datasets import register_datasets
from deepent.config import add_deepent_config

def visualize(predictor, data, dataset):
    fig, axes = plt.subplots(2,6,figsize=(40,15))
    for ax in axes.ravel():
        img = cv2.imread(random.sample(data, 1)[0]["file_name"])
        outputs = predictor(img)
        visualizer = Visualizer(img, metadata=MetadataCatalog.get(dataset+'_train'), instance_mode=ColorMode.IMAGE_BW)
        vis = visualizer.draw_dataset_dict(outputs["instances"])
        ax.imshow(vis.get_image())

def setup(args, model):
    cfg = get_cfg()
    add_deepent_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args, dataset, model):
    cfg = setup(args, model)
    predictor = DefaultPredictor(cfg)
    register_datasets('/home/ubuntu/tiled-data/')
    with open(os.path.join('/home/ubuntu/tiled-data/train/', dataset+'/segs.json'), 'r') as f:
        data = json.load(f)
    visualize(predictor, data, dataset)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    dataset = 'Kelowna'
    model = '/home/ubuntu/drone-tree-id/output/model_0034999.pth'
    main(args, dataset, model)
