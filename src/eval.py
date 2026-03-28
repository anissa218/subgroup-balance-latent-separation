from typing import Any, Dict, List, Tuple

import hydra
import rootutils
import os
import torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import pandas as pd

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

torch.set_float32_matmul_precision('high')

log = RankedLogger(__name__, rank_zero_only=True)

@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    log.info("Generating predictions")
    predict_dataloader = datamodule.test_dataloader()
    preds = model.get_test_outputs()
    # preds = trainer.predict(model=model, dataloaders=predict_dataloader, ckpt_path=cfg.ckpt_path)
    all_preds = torch.cat([batch["preds"] for batch in preds]).cpu().numpy()
    all_targets = torch.cat([batch["targets"] for batch in preds]).cpu().numpy()

    if "ids" in preds[0]:
        all_ids = [id for batch in preds for id in batch["ids"]]  # flatten list of tuples
        if "logits" in preds[0]:
            all_logits = torch.cat([batch["logits"] for batch in preds]).cpu().numpy()
            df = pd.DataFrame({"id": all_ids, "target": all_targets, "prediction": all_preds, "logits_0": all_logits[:, 0], "logits_1": all_logits[:, 1]}) # not all models save logits
        else:
            df = pd.DataFrame({"id": all_ids, "target": all_targets, "prediction": all_preds})
    else:
        df = pd.DataFrame({"target": all_targets, "prediction": all_preds})
        
    df.to_csv(os.path.join(cfg.paths.output_dir,"preds.csv"), index=False)

    if "features" in preds[0]:
        log.info("Saving features!")
        all_features = torch.cat([batch["features"] for batch in preds]).cpu()
        torch.save(all_features,os.path.join(cfg.paths.output_dir,"features.pt"))

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
