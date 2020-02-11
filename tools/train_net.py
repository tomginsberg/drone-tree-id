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
    def test(cls, cfg, model, evaluators=None):
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
                for task, res in results.items():
                    # Don't print "AP-category" metrics since they are usually not tracked.
                    wandb.log({f'{task}/{k}': v for k, v in res.items() if "-" not in k})

        if len(results) == 1:
            results = list(results.values())[0]
        return results


def setup(args):
    cfg = get_cfg()
    add_deepent_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
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
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    try:
        register_datasets('/home/ubuntu/RGBD-Tree-Segs')
    except FileNotFoundError:
        register_datasets('RGBD-Tree-Segs')

    args = default_argument_parser().parse_args()
    if args.eval_only:
        wandb.init(project="forest", name=args.opts[-2])
    wandb.init(project="forest")

    launch(main, args.num_gpus, args.num_machines, args.machine_rank, args.dist_url, args=(args,))
