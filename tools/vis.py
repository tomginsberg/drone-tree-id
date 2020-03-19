import argparse
import os
import random
from typing import List, Dict, Any

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import CfgNode

from deepent.config import add_deepent_config
from deepent.data.register_datasets import register_datasets

from tools.predictor import RGBDPredictor


def no_annotations_data_filter(dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter data that should not be visualized. I.e. tiles without annotations, boarder tiles, black tiles etc.

    Args:
        dicts: a list of data dictionaries obtained from a registered dataset
        Usually obtained from `list(DatasetCatalog.get(dataset))`

    Returns:
        object: a filtered list of data with annotations with at most half the length of the original list

    """
    return list(filter(lambda x: len(x['annotations']) > 0, dicts))[:len(dicts) // 2]


def visualize_comparison(predictor:RGBDPredictor, data: List[Dict[str, Any]], metadata: str, output: str, samples: int,
                         prefix: str,
                         data_filtering_function=lambda x: x[:len(x // 2)]):
    """
    Generate two plots side-by-side, one displaying the model's inference and the other the annotations of the data sample
    This function performs the aforementioned for up to `samples` number of samples depending on the filtering chosen

    Args:
        predictor: a predictor instance used for inference
        data: a list of data dictionaries obtained from a registered dataset, the length of which should compensate for
            the filtering function chosen
        metadata: the metadata associated with the dataset
        output: the path to store the images
        samples: how many samples to compare
        prefix: a common prefix used to name each sample visualization
        data_filtering_function: a function used to filter the data, default selects first half of data list

    """
    dicts = data_filtering_function(random.sample(data, 2 * samples))
    for dic in tqdm(dicts):
        fig, axes = plt.subplots(1, 2)
        ax = axes.ravel()
        img, predictions = get_predictions(predictor, dic["file_name"])
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


def visualize_many(predictor:RGBDPredictor, data: List[Dict[str, Any]], metadata: str, output: str, samples: int,
                   prefix: str):
    """
    Generate multiple inference plots in a single image
    This function performs the aforementioned for `samples` number of samples depending on the filtering chosen

    Args:
        predictor: a predictor instance used for inference
        data: a list of data dictionaries obtained from a registered dataset
        metadata: the metadata associated with the dataset
        output: the path to store the images
        samples: how many samples to compare
        prefix: a common prefix used to name each sample visualization

    """
    dicts = random.sample(data, samples)
    for dic in tqdm(dicts):
        # Define layout of plots, rows by columns
        fig, axes = plt.subplots(2, 6)
        for ax in axes.ravel():
            img, predictions = get_predictions(predictor, dic["file_name"])
            visualizer = Visualizer(img, metadata=metadata)
            vis = visualizer.draw_instance_predictions(predictions["instances"].to("cpu")).get_image()
            title = prefix + os.path.basename(dic["file_name"])
            ax.set_axis_off()
            ax.imshow(vis)
        os.makedirs(output, exist_ok=True)
        plt.savefig(os.path.join(output, title))
        plt.close()


def visualize_single(predictor:RGBDPredictor, data: List[Dict[str, Any]], metadata: str, output: str, samples: int,
                     prefix: str):
    """
    Generate single inference plots
    This function performs the aforementioned for `samples` number of samples depending on the filtering chosen

    Args:
        predictor: a predictor instance used for inference
        data: a list of data dictionaries obtained from a registered dataset
        metadata: the metadata associated with the dataset
        output: the path to store the images
        samples: how many samples to compare
        prefix: a common prefix used to name each sample visualization

    """
    dicts = random.sample(data, samples)
    for dic in tqdm(dicts):
        img, predictions = get_predictions(predictor, dic["file_name"])
        visualizer = Visualizer(img, metadata=metadata)
        vis = visualizer.draw_instance_predictions(predictions["instances"].to("cpu")).get_image()
        title = prefix + os.path.basename(dic["file_name"])
        plt.title(title)
        plt.axis('off')
        plt.imshow(vis)
        os.makedirs(output, exist_ok=True)
        plt.savefig(os.path.join(output, title))
        plt.close()


def get_predictions(predictor: RGBDPredictor, img_path: str):
    """
    Obtain predictions for RGB and RGBD models

    Args:
        predictor: predictor used for inference
        img_path: path to image

    Returns: RGB image and prediction dictionary
    """
    # Handle loading in RGBD or RGB image for any predictor
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    predictions = predictor(img)
    img = img[:,:,:3]
    return img, predictions


def setup(args) -> CfgNode:
    """
    Setup config file for predictor

    Args: see arg parser

    Returns: the compiled config

    """
    # load a black config
    cfg = get_cfg()
    # load custom configs
    add_deepent_config(cfg)
    # merge in configs from base config file
    cfg.merge_from_file(args.config_file)
    opts = args.opts
    if args.seed:
        opts.append("SEED")
        opts.append(args.seed)
    cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = args.model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold
    cfg.freeze()
    return cfg


def main(args):
    """
    Visualizes inference on a random sample of data from specified datatsets.
    Visualizations can be single or multiple inference or single comparisons with annotations.

    Args:
        args: see arg parser
    """
    register_datasets(args.data_path)
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
            visualize_comparison(predictor, data, metadata, output, args.samples, prefix,
                                 data_filtering_function=no_annotations_data_filter)
        else:
            visualize_many(predictor, data, metadata, output, args.samples, prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes instance predictions."
    )
    parser.add_argument("--data_path", required=True, type=str, help="Path to dataset to register")
    parser.add_argument("--model", required=True, type=str, help="Path to model weights")
    parser.add_argument("--config-file", required=True, help="Path to config")
    parser.add_argument("--dataset", help="name of the dataset from which to sample, can be multiple", type=str,
                        nargs='+', default="CPT2a-n_test")
    parser.add_argument("--threshold", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--opts", default=[], type=list, help="additional config options")
    parser.add_argument("--samples", default=1, type=int, help="number of visualizations to produce")
    parser.add_argument("--output", default=None, type=str, help="where to store visualizations")
    parser.add_argument("--seed", default=-1, type=int,
                        help="random seed to keep data sampling consistent for multiple visualizations, -1 for auto "
                             "generated seed")
    parser.add_argument("--type", default='many', type=str, choices=['single', 'many', 'comparison'],
                        help="type of plot, single, comparison and many")
    args = parser.parse_args()
    main(args)
