import os
import random

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, default_argument_parser, default_setup
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

from deepent.data_processing import register_datatsets

def visualize(predictor, data, dataset):
    fig, axes = plt.subplots(2,6,figsize=(40,15))
    for ax in axes.ravel():
        img = cv2.imread(random.sample(data, 1)["file_name"])
        outputs = predictor(img)
        visualizer = Visualizer(img, metadata=MetadataCatalog.get(dataset), instance_mode=ColorMode.IMAGE_BW)
        vis = visualizer.draw_dataset_dict(outputs["instances"].to("cpu"))
        ax.imshow(vis.get_image())

def setup(args):
    cfg = get_cfg()
    add_deepent_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, args.model)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.threshold
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    '''
    config_file -> saved weights
    threshold -> score threshold
    dataset -> where to get images from
    '''
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)
    register_datatsets()
    dataset_path = f'FYBRData/test/{args.dataset}'
    with open(f'{dataset_path}/segs.json', 'r') as f:
        data = json.load(f)
    visualize(predictor, data, args.dataset)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
