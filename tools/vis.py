import os
import random
import matplotlib.pyplot as plt
import cv2
import argparse

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, default_setup
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from deepent.data.register_datasets import register_datasets
from deepent.config import add_deepent_config

def visualize_comparison(predictor, data, metadata, output, samples, prefix):
    dicts = random.sample(data, samples)
    for dic in dicts:
        fig, axes = plt.subplots(1,2)
        ax = axes.ravel()
        img = cv2.imread(dic["file_name"])
        predictions = predictor(img)
        visualizer = Visualizer(img, metadata=metadata, instance_mode=ColorMode.IMAGE_BW)
        vis = visualizer.draw_instance_predictions(predictions["instances"].to("cpu")).get_image()
        ax[0].set_title('Prediction')
        ax[0].imshow(vis)
        visualizer2 = Visualizer(img, metadata=metadata, instance_mode=ColorMode.IMAGE_BW)
        vis2 = visualizer.draw_dataset_dict(dic).get_image()
        ax[1].set_title('Annotation')
        ax[1].imshow(vis2)
        os.makedirs(output, exist_ok=True)
        plt.savefig(os.path.join(output, prefix+os.path.basename(dic["file_name"])))

def visualize_many(predictor, data, metadata, output, samples, prefix):
    for _ in range(samples):
        fig, axes = plt.subplots(2,6)
        for ax in axes.ravel():
            dic = random.sample(data, 1)[0]
            img = cv2.imread(dic["file_name"])
            predictions = predictor(img)
            visualizer = Visualizer(img, metadata=metadata, instance_mode=ColorMode.IMAGE_BW)
            vis = visualizer.draw_instance_predictions(predictions["instances"].to("cpu")).get_image()
            title = prefix+os.path.basename(dic["file_name"])
            ax.set_title(title)
            ax.imshow(vis)
        os.makedirs(output, exist_ok=True)
        plt.savefig(os.path.join(output, title))

def visualize_single(predictor, data, metadata, output, sample, prefix):
    for _ in range(samples):
        dic = random.sample(data, 1)[0]
        img = cv2.imread(dic["file_name"])
        predictions = predictor(img)
        visualizer = Visualizer(img, metadata=metadata, instance_mode=ColorMode.IMAGE_BW)
        vis = visualizer.draw_instance_predictions(predictions["instances"].to("cpu")).get_image()
        title = prefix+os.path.basename(dic["file_name"])
        plt.title(title)
        plt.imshow(vis)
        os.makedirs(output, exist_ok=True)
        plt.savefig(os.path.join(output, title))

def setup(args):
    cfg = get_cfg()
    add_deepent_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.model 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold 
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)
    register_datasets(f'/home/ubuntu/RGBD-Tree-Segs/')
    data = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    output = os.path.join(os.path.realpath(cfg.OUTPUT_DIR if args.output is None else args.output), args.dataset) 
    prefix = os.path.basename(args.model).split('.')[0]+'_thresh'+str(args.threshold)+'_'
    if args.type == 'single':
        return visualize_single(predictor, data, metadata, output, args.samples,  prefix)
    elif args.type == 'comparison':
        return visualize_comparison(predictor, data, metadata, output, args.samples, prefix)
    return visualize_many(predictor, data, metadata, output, args.samples, prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes instance predictions."
    )
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--config-file", required=True, help="Path to config")
    parser.add_argument("--dataset", help="name of the dataset", type=str, default="CPT2a-n_test")
    parser.add_argument("--threshold", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--opts", default=[], type=list, help="additional options")
    parser.add_argument("--samples", default=1, type=int, help="number of sample visualizations to produce")
    parser.add_argument("--output", default=None, type=str, help="output directory")
    parser.add_argument("--type", default='many', type=str, choices=['single', 'many', 'comparison'], help="type of plot, single, comparison and many")
    args = parser.parse_args()
    main(args)
