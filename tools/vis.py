import argparse
import os
import random

import cv2
import matplotlib.pyplot as plt

from deepent.config import add_deepent_config
from deepent.data.register_datasets import register_datasets
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import default_setup
from detectron2.utils.visualizer import Visualizer, ColorMode

from deepent.data.register_datasets import register_datasets
from deepent.config import add_deepent_config
from tools.predictor import RGBDPredictor

def visualize_comparison(predictor, data, metadata, output, samples, prefix):
    dicts = random.sample(data, samples)
    for dic in dicts:
        fig, axes = plt.subplots(1, 2)
        ax = axes.ravel()
        rgba = cv2.imread(dic["file_name"], cv2.IMREAD_UNCHANGED)
        img = cv2.imread(dic["file_name"])
        try:
            predictions = predictor(rgba)
        except:
            predictions = predictor(img)
        visualizer = Visualizer(img, metadata=metadata)
        vis = visualizer.draw_instance_predictions(predictions["instances"].to("cpu")).get_image()
        ax[0].set_title('Prediction')
        ax[0].set_axis_off()
        ax[0].imshow(vis)
        visualizer2 = Visualizer(img, metadata=metadata)
        vis2 = visualizer2.draw_dataset_dict(dic).get_image()
        ax[1].set_title('Annotation')
        ax[1].set_axis_off()
        ax[1].imshow(vis2)
        os.makedirs(output, exist_ok=True)
        plt.savefig(os.path.join(output, prefix + os.path.basename(dic["file_name"])))
        plt.close()


def visualize_many(predictor, data, metadata, output, samples, prefix):
    dicts = random.sample(data, samples)
    for dic in dicts:
        fig, axes = plt.subplots(2,6)
        for ax in axes.ravel():
            img = cv2.imread(dic["file_name"])
            rgba = cv2.imread(dic["file_name"], cv2.IMREAD_UNCHANGED)
            predictions = predictor(rgba)
            visualizer = Visualizer(img, metadata=metadata)
            vis = visualizer.draw_instance_predictions(predictions["instances"].to("cpu")).get_image()
            title = prefix + os.path.basename(dic["file_name"])
            ax.set_axis_off()
            ax.imshow(vis)
        os.makedirs(output, exist_ok=True)
        plt.savefig(os.path.join(output, title))
        plt.close()


def visualize_single(predictor, data, metadata, output, samples, prefix):
    dicts = random.sample(data, samples)
    for dic in dicts:
        rgba = cv2.imread(dic["file_name"], cv2.IMREAD_UNCHANGED)
        img = cv2.imread(dic["file_name"])
        predictions = predictor(rgba)
        visualizer = Visualizer(img, metadata=metadata)
        vis = visualizer.draw_instance_predictions(predictions["instances"].to("cpu")).get_image()
        title = prefix + os.path.basename(dic["file_name"])
        plt.title(title)
        plt.axis('off')
        plt.imshow(vis)
        os.makedirs(output, exist_ok=True)
        plt.savefig(os.path.join(output, title))
        plt.close()


def setup(args):
    cfg = get_cfg()
    add_deepent_config(cfg)
    cfg.merge_from_file(args.config_file)
    opts = args.opts
    if args.seed:
        opts.append("SEED")
        opts.append(-1)
    cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = args.model 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold 
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    register_datasets(f'/home/ubuntu/RGBD-Tree-Segs/')
    cfg = setup(args)
    predictor = RGBDPredictor(cfg)
    for dataset in args.dataset:
        data = list(DatasetCatalog.get(dataset))
        metadata = MetadataCatalog.get(dataset)
        output = os.path.join(os.path.realpath(cfg.OUTPUT_DIR if args.output is None else args.output), dataset)
        prefix = os.path.basename(args.model).split('.')[0] + '_thresh' + str(args.threshold) + '_'
        if args.type == 'single':
            visualize_single(predictor, data, metadata, output, args.samples, prefix)
        elif args.type == 'comparison':
            visualize_comparison(predictor, data, metadata, output, args.samples, prefix)
        else:
            visualize_many(predictor, data, metadata, output, args.samples, prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes instance predictions."
    )
    parser.add_argument("--model", required=True, type=str, help="Path to model weights")
    parser.add_argument("--config-file", required=True, help="Path to config")
    parser.add_argument("--dataset", help="name of the dataset", type=str, nargs='+', default="CPT2a-n_test")
    parser.add_argument("--threshold", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--opts", default=[], type=list, help="additional options")
    parser.add_argument("--samples", default=1, type=int, help="number of sample visualizations to produce")
    parser.add_argument("--output", default=None, type=str, help="output directory")
    parser.add_argument("--seed", default=False, type=bool, help="use random random seed")
    parser.add_argument("--type", default='many', type=str, choices=['single', 'many', 'comparison'],
                        help="type of plot, single, comparison and many")
    args = parser.parse_args()
    main(args)
