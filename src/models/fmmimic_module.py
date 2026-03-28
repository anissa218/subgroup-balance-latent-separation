from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from functools import partial

class FMMIMICLitModule(LightningModule):
    """Example of a `LightningModule` for MIMIC classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        loss_weight_class_0: float = 1.0,
    ) -> None:
        """Initialize a `MIMICLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.loss_weights = torch.tensor([self.hparams.loss_weight_class_0, 1.0])
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.loss_weights)

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="binary", num_classes=2)
        self.val_acc = Accuracy(task="binary", num_classes=2)
        self.test_acc = Accuracy(task="binary", num_classes=2)

        self.train_bal_acc = Accuracy(task="multiclass", num_classes=2, average="macro")
        self.val_bal_acc = Accuracy(task="multiclass", num_classes=2, average="macro")
        self.test_bal_acc = Accuracy(task="multiclass", num_classes=2, average="macro")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_bal_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()
        self.val_bal_acc.reset()
        self.val_bal_acc_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        _, logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels and dicom_id.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        x, y, _ = batch
        loss, preds, targets = self.model_step((x,y)) # don't need to pass dicom_id

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=False, on_epoch=True, prog_bar=False)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels and dicom_id.
        :param batch_idx: The index of the current batch.
        """
        x, y, _ = batch
        loss, preds, targets = self.model_step((x,y))

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_bal_acc(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/bal_acc", self.val_bal_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        bal_acc = self.val_bal_acc.compute()  # get current val balanced acc
        self.val_bal_acc_best(bal_acc)  # update best so far val balanced acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/bal_acc_best", self.val_bal_acc_best.compute(), sync_dist=True, prog_bar=True)
    
    def on_test_epoch_start(self) -> None:
        self.test_outputs = []

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels and dicom_id.
        :param batch_idx: The index of the current batch.

        :return: A dict with the model predictions and with the ground truth target labels.
        """
        x, y, dicom_id = batch
        features, logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        targets = y

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_bal_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/bal_acc", self.test_bal_acc, on_step=False, on_epoch=True, prog_bar=True)

        output = {
            "ids": dicom_id,
            "preds": preds.detach().cpu(),
            "targets": y.detach().cpu(),
            "logits": logits.detach().cpu(),
            "features": features.detach().cpu()
        }
        self.test_outputs.append(output)

    def on_test_epoch_end(self) -> None:
        """Do nothing special. Required only if you do not return from `test_epoch_end`."""
        pass

    def get_test_outputs(self):
        return self.test_outputs

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

        self.loss_weights = self.loss_weights.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.loss_weights)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}

        scheduler_cfg = self.hparams.scheduler

        # for sequential schedulers (to do warmup + cosine)
        if isinstance(scheduler_cfg, partial) and scheduler_cfg.func == torch.optim.lr_scheduler.SequentialLR:
            schedulers_partials = scheduler_cfg.keywords["schedulers"]
            milestones = scheduler_cfg.keywords["milestones"]

            # instantiate each inner scheduler with optimizer
            schedulers = [sched(optimizer=optimizer) for sched in schedulers_partials]

            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=schedulers,
                milestones=milestones,
            )

        # standard single scheduler
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

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A dict with the model predictions and with the ground truth target labels.
        """
        x, y, dicom_id = batch
        _,logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        return {"ids": dicom_id, "preds": preds, "targets": y, "logits": logits}



if __name__ == "__main__":
    _ = FMMIMICLitModule(None, None, None, None)
