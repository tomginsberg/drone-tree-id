import os
import shutil
from glob import glob

import fire

from deepent.config import add_deepent_config
from deepent.data.build_tiled_json_dataset import DataTiler, ignore_black_tiles, remove_no_annotations, \
    remove_small_segment_coverage, remove_small_bboxes
from deepent.data.untiler import Untiler
from detectron2.config import get_cfg
from tools.predictor import RGBDPredictor

PREDICTORS = {'sequoia': {'model': 'output/baseline_25_01_2020/model_0054999.pth'},
              'redwood': {'model': 'output/fuse_long/model_final.pth'},
              'fuselong90': {'model': 'output/fuse_long/model_0089999.pth'},
              'fuselong75': {'model': 'output/fuse_long/model_0074999.pth'},
              'fuselong60': {'model': 'output/fuse_long/model_0059999.pth'},
              'fuselong45': {'model': 'output/fuse_long/model_0044999.pth'},
              'fuselong30': {'model': 'output/fuse_long/model_0029999.pth'},
              'fuselong15': {'model': 'output/fuse_long/model_0014999.pth'},
              'fuselong5': {'model': 'output/fuse_long/model_0004999.pth'},
              'fuse90': {'model': 'output/baseline_fuse_07_02_2020/model_0089999.pth'},
              'fuse75': {'model': 'output/baseline_fuse_07_02_2020/model_0074999.pth'},
              'fuse60': {'model': 'output/baseline_fuse_07_02_2020/model_0059999.pth'},
              'fuse45': {'model': 'output/baseline_fuse_07_02_2020/model_0044999.pth'},
              'fuse30': {'model': 'output/baseline_fuse_07_02_2020/model_0029999.pth'},
              'fuse15': {'model': 'output/baseline_fuse_07_02_2020/model_0014999.pth'},
              'fuse5': {'model': 'output/baseline_fuse_07_02_2020/model_0004999.pth'},
              'fuselateral': {'model': 'output/fuse_lateral/model_final.pth'},
              'fuselateral60': {'model': 'output/fuse_lateral/model_0059999.pth'},
              'fuselateral45': {'model': 'output/fuse_lateral/model_0044999.pth'},
              'fuselateral15': {'model': 'output/fuse_lateral/model_0014999.pth'},
              'fuselateral5': {'model': 'output/fuse_lateral/model_0004999.pth'},
              'fuse24_90': {'model': 'output/fuse_24/model_0089999.pth'},
              'fuse24_75': {'model': 'output/fuse_24/model_0074999.pth'},
              'fuse24_60': {'model': 'output/fuse_24/model_0059999.pth'},
              'fuse24_45': {'model': 'output/fuse_24/model_0044999.pth'},
              'fuse24_15': {'model': 'output/fuse_24/model_0014999.pth'},
              'fuse24_5': {'model': 'output/fuse_24/model_0004999.pth'},
              'rgbd40': {'model': 'output/rgbd/model_0039999.pth'},
              'rgbd30': {'model': 'output/rgbd/model_0029999.pth'},
              'rgbd15': {'model': 'output/rgbd/model_0014999.pth'},
              'rgbd5': {'model': 'output/rgbd/model_0004999.pth'},
              'cpts45': {'model': 'output/cpts/model_0044999.pth'},
              'cpts30': {'model': 'output/cpts/model_0029999.pth'},
              'cpts15': {'model': 'output/cpts/model_0014999.pth'},
              'cpts5': {'model': 'output/cpts/model_0004999.pth'},
              'rgb45': {'model': 'output/baseline_25_01_2020/model_0044999.pth'},
              'rgb30': {'model': 'output/baseline_25_01_2020/model_0029999.pth'},
              'rgb15': {'model': 'output/baseline_25_01_2020/model_0014999.pth'},
              'rgb5': {'model': 'output/baseline_25_01_2020/model_0004999.pth'}
              }


def run_description(predictors):
    for i, ensemble in enumerate(predictors):
        print(f'Run {i + 1}: {"+".join(ensemble)}')
        for k, model in enumerate(ensemble):
            try:
                data = PREDICTORS[model]
                print(
                    f'Model {k + 1}: Name: {model}'
                    f' \n\t Weights: {data["model"]}\n')
            except KeyError:
                print(f'Model \'{model}\' not in Model Zoo!')


def get_predictor(model, confidence):
    cfg = get_cfg()
    add_deepent_config(cfg)
    model_config = os.path.join(os.path.dirname(model), 'config.yaml')
    cfg.merge_from_file(os.path.realpath(model_config))
    cfg.MODEL.WEIGHTS = os.path.realpath(model)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    cfg.freeze()
    return RGBDPredictor(cfg)


class ProjectManager:
    """
        Creates a manger to handle data pre-processing, predictions, and post-processing
        on a dataset of Ortho, CHM pairs.

        :param data:
        Option 1:
        File path to a folder containing multiple datasets of Ortho+CHM pairs.
        Option 2:
        Direct path to a single ortho and CHM.
        Option 3 (Advanced):
        Direct path to inference tiles generated using deepent/data/build_tiled_dataset.py.
        For this option, the offsets.json file must be located directly at the given location.
        Tiles will not be deleted after run.


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
                 duplicate_tol: float = .8,
                 min_area: float = 4,
                 use_generated_tiles: bool = False,
                 retain_tiles: bool = False):
        self.is_tile_path = len(glob(os.path.join(data, 'offsets.json'))) == 1

        self.path_to_raw_data = os.path.realpath(data)

        self.multiple_datasets = len(
            glob(os.path.join(self.path_to_raw_data, '*.png' if self.is_tile_path else '*.tif'))) == 0

        assert (not self.multiple_datasets and self.is_tile_path), 'A custom tile path with multiple ' \
                                                                   'datasets is not currently supported'

        self.confidence, self.duplicate_tol, self.min_area = confidence, duplicate_tol, min_area
        if isinstance(predictors, tuple):
            predictors = ','.join(predictors)

        combos = [[y.strip() for y in x.split('+')] for x in predictors.split(',')]
        run_description(combos)
        self.predictor_combos = [
            ('-'.join(combo), [get_predictor(**PREDICTORS[model], confidence=self.confidence) for model in combo]) for
            combo in combos]

        if shapefile_location is None:
            self.output = os.path.realpath('shapefiles')
        else:
            self.output = os.path.realpath(shapefile_location)

        if not self.is_tile_path:
            self.data_tiler = DataTiler(self.path_to_raw_data, os.path.join(self.path_to_raw_data, 'tmp'),
                                        vertical_overlay=320,
                                        horizontal_overlay=320, dataset_regex=datasets.strip().split(','),
                                        cleanup_on_init=not use_generated_tiles, create_inference_tiles=True)

            if not use_generated_tiles:
                self.prepare_inference_set()

        self.run_predictions()

        if not self.is_tile_path:
            if use_generated_tiles:
                retain_tiles = True
            if not retain_tiles:
                self.clean_tiles()

    def prepare_inference_set(self):
        self.data_tiler.tile_dataset(
            tile_filtering_function=ignore_black_tiles(thresh=.99),
            annotation_filtering_function=lambda an: remove_no_annotations(an) and remove_small_segment_coverage()(an),
            bbox_filtering_function=remove_small_bboxes(1000))

    def run_predictions(self):
        # Case 1, user has passed a path directly to tiles
        if self.is_tile_path:
            dataset_name = os.path.basename(self.path_to_raw_data)
            for predictor_name, predictors in self.predictor_combos:
                Untiler(predictors).predict_and_untile(self.path_to_raw_data,
                                                       os.path.join(self.output, dataset_name,
                                                                    f'{dataset_name}_{predictor_name}'),
                                                       duplicate_tol=self.duplicate_tol, min_area=self.min_area)

        # Case 2, user has passed a path to untiled data
        for dataset_name, dataset_path in zip(self.data_tiler.dataset_names, self.data_tiler.dataset_input_paths):
            if self.multiple_datasets:
                # If a full data collection is passed the tile directory will be in the given path
                dataset_path = self.path_to_raw_data
            for predictor_name, predictors in self.predictor_combos:
                Untiler(predictors).predict_and_untile(os.path.join(dataset_path, 'tmp', 'tiles', dataset_name),
                                                       os.path.join(self.output, dataset_name,
                                                                    f'{dataset_name}_{predictor_name}'),
                                                       duplicate_tol=self.duplicate_tol, min_area=self.min_area)

    def clean_tiles(self):
        if self.multiple_datasets:
            shutil.rmtree(os.path.join(self.path_to_raw_data, 'tmp'))
        else:
            for dp in self.data_tiler.dataset_input_paths:
                shutil.rmtree(os.path.join(dp, 'tmp'))


if __name__ == '__main__':
    fire.Fire(ProjectManager)
