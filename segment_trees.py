import os
import shutil
from glob import glob

import fire

from deepent.config import add_deepent_config
from deepent.data.build_tiled_json_dataset import DataTiler, ignore_black_tiles, remove_no_annotations, \
    remove_small_segment_coverage, remove_small_bboxes
from deepent.data.untiler import Untiler
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tools.predictor import RGBDPredictor

PREDICTORS = {'sequoia': {'config_file': 'configs/deepent_rcnn_R_50_FPN.yaml',
                          'model': 'output/baseline_25_01_2020/model_0054999.pth', 'predictor': DefaultPredictor},
              'redwood': {'config_file': 'configs/deepent_fuse_rcnn_R_50_FPN.yaml',
                          'model': 'output/fuse_long/model_final.pth', 'predictor': RGBDPredictor},
              'fuselong30': {'config_file': 'configs/deepent_fuse_rcnn_R_50_FPN.yaml',
                             'model': 'output/fuse_long/model_0029999l.pth', 'predictor': RGBDPredictor},
              'fuselong60': {'config_file': 'configs/deepent_fuse_rcnn_R_50_FPN.yaml',
                             'model': 'output/fuse_long/model_0059999.pth', 'predictor': RGBDPredictor},
              'fuselong90': {'config_file': 'configs/deepent_fuse_rcnn_R_50_FPN.yaml',
                             'model': 'output/fuse_long/model_0089999.pth', 'predictor': RGBDPredictor},
              'elder': {'config_file': 'configs/deepent_fuse_rcnn_R_50_FPN.yaml',
                        'model': 'output/baseline_fuse_07_02_2020/model_0069999.pth', 'predictor': RGBDPredictor},
              'oak': {'config_file': 'configs/deepent_fuse_rcnn_R_50_FPN.yaml',
                      'model': 'output/baseline_fuse_07_02_2020/model_0089999.pth', 'predictor': RGBDPredictor}
              }


def run_description(predictors):
    for i, ensemble in enumerate(predictors):
        print(f'Run {i + 1}: {"+".join(ensemble)}')
        for k, model in enumerate(ensemble):
            try:

                data = PREDICTORS[model]
                print(
                    f'Model {k + 1}: Name: {model}, \n\tConfig: {data["config_file"]}'
                    f' \n\tWeights: {data["model"]}\n\tRGBD Model {data["predictor"] == RGBDPredictor}\n')
            except KeyError:
                print(f'Model \'{model}\' not in Model Zoo!')


class ProjectManager:
    """
        Creates a manger to handle data pre-processing, predictions, and post-processing
        on a dataset of Ortho, CHM pairs.
        :param data: File path to a folder containing multiple datasets of Ortho+CHM pairs,
        or the direct path to a single ortho and CHM
        :param shapefile_location: File path to where the output shapefiles should be written.
        Default drone-tree-id/shapefiles
        :param predictors: A string of predictor ensembles separated by commas. Predictors are specified by their names
        in the MODELZOO, ensembles are denoted with the '+' symbol.
        Default 'redwood+sequoia'
        Other examples: 'redwood, sequoia, redwood+sequoia', 'sequoia, redwood+sequoia'
        :param datasets: If multiple datasets exist in --data, then this flag may be passed as a comma separated list of
        regular expressions to indicate which datasets should be kept
        Default '*'
        Examples 'CPT*, Kelowna, AM*'
        :param confidence: A value in [0, 1]. Confidence threshold for keeping a prediction of a single tree.
        Default .5
        :param duplicate_tol: A value in [0, 1]. Threshold for keeping a new prediction that is similar to an existing one
        i.e a new polygon will be written if it is less then (100*duplicate_tol)% similar to all others
        Default .85
        :param min_area: A float value. Minimum polygon area in m^2 to keep in final prediction.
        Default 4
        :param use_generated_tiles: Look for tiles have have been previously generated
        Default False
        :param retain_tiles: Keep generated tiles after a run
        Default False
    """

    def __init__(self, data: str, shapefile_location: str = None, predictors='sequoia', datasets: str = '*',
                 confidence: float = .5,
                 duplicate_tol: float = .85,
                 min_area: float = 5,
                 use_generated_tiles: bool = False,
                 retain_tiles: bool = False):
        self.path_to_raw_data = os.path.realpath(data)
        self.multiple_datasets = len(glob(os.path.join(self.path_to_raw_data, '*.tif'))) == 0
        self.confidence, self.duplicate_tol, self.min_area = confidence, duplicate_tol, min_area

        combos = [[y.strip() for y in x.split('+')] for x in predictors.split(',')]
        run_description(combos)
        self.predictor_combos = [('-'.join(combo), [self.get_predictor(**PREDICTORS[model]) for model in combo]) for
                                 combo in combos]

        if shapefile_location is None:
            self.output = os.path.realpath('shapefiles')
        else:
            self.output = os.path.realpath(shapefile_location)

        self.data_tiler = DataTiler(self.path_to_raw_data, os.path.join(self.path_to_raw_data, 'tmp'),
                                    vertical_overlay=400,
                                    horizontal_overlay=400, dataset_regex=datasets.strip().split(','),
                                    cleanup_on_init=not use_generated_tiles)

        if not use_generated_tiles:
            self.prepare_inference_set()

        self.run_predictions()

        if not retain_tiles:
            self.clean_tiles()

    def get_predictor(self, config_file, model, predictor):
        cfg = get_cfg()
        add_deepent_config(cfg)
        cfg.merge_from_file(os.path.realpath(config_file))
        cfg.MODEL.WEIGHTS = os.path.realpath(model)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence
        cfg.freeze()

        return predictor(cfg)

    def prepare_inference_set(self):
        self.data_tiler.tile_dataset(
            tile_filtering_function=ignore_black_tiles(thresh=.99),
            annotation_filtering_function=lambda an: remove_no_annotations(an) and remove_small_segment_coverage()(an),
            bbox_filtering_function=remove_small_bboxes(1000),
            no_train=True)

    def run_predictions(self):
        for dataset_name, dataset_path in zip(self.data_tiler.dataset_names, self.data_tiler.dataset_input_paths):
            if self.multiple_datasets:
                # If a full data collection is passed the tile directory will be in the given path
                dataset_path = self.path_to_raw_data
            for predictor_name, predictors in self.predictor_combos:
                Untiler(predictors).predict_and_untile(os.path.join(dataset_path, 'tmp', 'tiles', dataset_name),
                                                       os.path.join(self.output, dataset_name,
                                                                    f'{dataset_name}_{predictor_name}'))

    def clean_tiles(self):
        if self.multiple_datasets:
            shutil.rmtree(os.path.join(self.path_to_raw_data, 'tmp'))
        else:
            for dp in self.data_tiler.dataset_input_paths:
                shutil.rmtree(os.path.join(dp, 'tmp'))


if __name__ == '__main__':
    fire.Fire(ProjectManager)
