"""LightningModule for SigLIP PEFT fine-tuning.

Identical to FMMIMICLitModule except configure_optimizers filters to trainable
parameters only (important for LoRA where most params are frozen).
"""

from typing import Any, Dict, Tuple
from functools import partial

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy


class SigLIPPEFTLitModule(LightningModule):
    """LightningModule for fine-tuning SigLIP with LoRA PEFT.

    Batch format expected: (pixel_values, label, dicom_id)  — same as other MIMIC modules.
    net.forward(pixel_values) must return (features, logits)  — same as other MIMIC nets.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        loss_weight_class_0: float = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net

        self.loss_weights = torch.tensor([self.hparams.loss_weight_class_0, 1.0])
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.loss_weights)

        self.train_acc = Accuracy(task="binary", num_classes=2)
        self.val_acc = Accuracy(task="binary", num_classes=2)
        self.test_acc = Accuracy(task="binary", num_classes=2)

        self.train_bal_acc = Accuracy(task="multiclass", num_classes=2, average="macro")
        self.val_bal_acc = Accuracy(task="multiclass", num_classes=2, average="macro")
        self.test_bal_acc = Accuracy(task="multiclass", num_classes=2, average="macro")

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()
        self.val_bal_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_bal_acc.reset()
        self.val_bal_acc_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        _, logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        x, y, _ = batch
        loss, preds, targets = self.model_step((x, y))
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=False, on_epoch=True, prog_bar=False)

    def validation_step(self, batch: Tuple, batch_idx: int) -> None:
        x, y, _ = batch
        loss, preds, targets = self.model_step((x, y))
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_bal_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/bal_acc", self.val_bal_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        bal_acc = self.val_bal_acc.compute()
        self.val_bal_acc_best(bal_acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/bal_acc_best", self.val_bal_acc_best.compute(), sync_dist=True, prog_bar=True)

    def on_test_epoch_start(self) -> None:
        self.test_outputs = []

    def test_step(self, batch: Tuple, batch_idx: int) -> None:
        x, y, dicom_id = batch
        features, logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_loss(loss)
        self.test_acc(preds, y)
        self.test_bal_acc(preds, y)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/bal_acc", self.test_bal_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.test_outputs.append({
            "ids": dicom_id,
            "preds": preds.detach().cpu(),
            "targets": y.detach().cpu(),
            "logits": logits.detach().cpu(),
            "features": features.detach().cpu(),
        })

    def on_test_epoch_end(self) -> None:
        pass

    def get_test_outputs(self):
        return self.test_outputs

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
        self.loss_weights = self.loss_weights.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.loss_weights)

    def configure_optimizers(self) -> Dict[str, Any]:
        # Only pass trainable parameters (LoRA adapters + classifier head).
        # Frozen backbone params have requires_grad=False and are excluded.
        trainable_params = [p for p in self.trainer.model.parameters() if p.requires_grad]
        optimizer = self.hparams.optimizer(params=trainable_params)

        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        scheduler_cfg = self.hparams.scheduler

        if isinstance(scheduler_cfg, partial) and scheduler_cfg.func == torch.optim.lr_scheduler.SequentialLR:
            schedulers_partials = scheduler_cfg.keywords["schedulers"]
            milestones = scheduler_cfg.keywords["milestones"]
            schedulers = [sched(optimizer=optimizer) for sched in schedulers_partials]
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=schedulers, milestones=milestones
            )
        else:
            scheduler = scheduler_cfg(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        x, y, dicom_id = batch
        _, logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        return {"ids": dicom_id, "preds": preds, "targets": y, "logits": logits}


if __name__ == "__main__":
    _ = SigLIPPEFTLitModule(None, None, None, None)
