import logging
import os
from collections import OrderedDict

import wandb

import detectron2.utils.comm as comm
from deepent.config import add_deepent_config
from deepent.data.register_datasets import register_datasets
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results, inference_on_dataset, \
    print_csv_format, DatasetEvaluator
from detectron2.utils.events import CommonMetricPrinter, JSONWriter
from detectron2.utils.logger import setup_logger
from tools.wandb_writer import WandbWriter

WANDB_PROJECT = "forest"


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = [COCOEvaluator(dataset_name, cfg, False, output_folder)]
        return DatasetEvaluators(evaluators)

    def build_writers(self):
        return [CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                WandbWriter()
                ]

    @classmethod
    def test(cls, cfg, model, evaluators=None, wandb_on=True):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            # for f in glob.glob(os.path.join('output', 'inference', '*')):
            #             #     print(f'Removing {f}')
            #             #     os.remove(f)
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
                logger = logging.getLogger(__name__)
                if wandb_on:
                    for task, res in results.items():
                        # Don't print "AP-category" metrics since they are usually not tracked.
                        wandb.log({f'{task}/{k}': v for k, v in res.items() if "-" not in k})

        if len(results) == 1:
            results = list(results.values())[0]
        return results


# custom test set: array of test set names to use in place of default (all)
def setup(args, custom_test_set=None):
    cfg = get_cfg()
    add_deepent_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if custom_test_set is not None:
        cfg.DATASETS.TEST = custom_test_set
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, name="deepent")
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model, wandb_on=True)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def start_logger(args):
    run_name = args.run_name
    if not args.no_log:
        if args.eval_only:
            if run_name is None:
                wandb.init(project=WANDB_PROJECT, name='_'.join(args.opts[-1].split('/')[-2:]) + '_eval')
            else:
                wandb.init(project=WANDB_PROJECT, name=run_name)
        else:
            if run_name is None:
                wandb.init(project=WANDB_PROJECT)
            else:
                wandb.init(project=WANDB_PROJECT, name=run_name)


if __name__ == "__main__":
    parser = default_argument_parser()

    parser.add_argument('--data_path', required=True, help="""Path to training/evaluation data.
        This dataset should be created using deepent/data/build_tiled_json_dataset.py
        with flag --create_inference_tiles True""", type=str)
    parser.add_argument('--run_name', default=None, help='Name for Wandb run.', type=str)
    parser.add_argument('--no_log', action="store_true", help='Enable to turn off Wandb logging for this run')

    args = parser.parse_args()

    start_logger(args)
    register_datasets(args.data_path)

    launch(main, args.num_gpus, args.num_machines, args.machine_rank, args.dist_url, args=(args,))
