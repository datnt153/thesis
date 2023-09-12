from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
import wandb

from pose_model import Model
from pose_data import MyDataset
from pose_model import Model

class CFG:
    # General
    debug = False 
    num_workers = 8
    gpus = 1

    # Training
    epochs = 10
    mixup = True
    backbone = "tf_efficientnetv2_m_in21k"
    val_check_interval = 1.0  # how many times we want to validate during an epoch
    check_val_every_n_epoch = 1
    gradient_clip_val = 1.0
    lr = 2e-4
    output_path = "output/deplot/v4"
    log_steps = 200
    batch_size = 2 
    use_wandb = True


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a.long()) + (1 - lam) * criterion(pred, y_b.long())

class PoseModel(pl.LightningModule):
    def __init__(self, model, hparams=None,  num_training_steps=None ):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(hparams)
        # Define your model structure
        self.model = model 
        # Define the loss function
        self.criterion = nn.CrossEntropyLoss()
        self.num_training_steps = num_training_steps


    def training_step(self, batch, batch_idx):
        # Training step
        img, pose, target = batch
        logits, target, target2, lam = self.model(img, pose, target)

        loss_func = self.criterion
        if CFG.mixup:
            loss = mixup_criterion(self.criterion, logits, target, target2, lam).mean()
        else:
            loss = loss_func(logits, target)

        # Logging metrics
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        img, pose, target = batch
        logits = self.model(img, pose)
        loss = self.criterion(logits, target.long())

        # Compute predictions
        preds = torch.argmax(logits, dim=1)

        # Append predictions and targets to lists
        self.preds.append(preds)
        self.targets.append(target)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def on_validation_start(self) -> None:
        # A list to store predictions and targets across batches
        self.preds = []
        self.targets = []

    def on_validation_epoch_end(self) -> None:
        # Concatenate all predictions and targets
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)

        # Compute accuracy
        acc = (preds == targets).float().mean()

        # Log metrics
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Reset the lists for the next epoch
        self.preds = []
        self.targets = []



    def configure_optimizers(self):
        # Define optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=CFG.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.epochs)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# view = "Right_side_window"
# setup dataset
fold = 0
event_names_with_background = ['Right_side_window', 'Dashboard', 'Rear_view', ]
for view in event_names_with_background:
    
    df_train = pd.read_csv(f"folds/fold_{fold}/train_{fold}.csv")
    # df_train = df_train.iloc[:100]
    df_val = pd.read_csv(f"folds/fold_{fold}/val_{fold}.csv")

    df_train = df_train[df_train["view"] == view]
    df_val = df_val[df_val["view"] == view]

    len_train = len(df_train)
    len_val = len(df_val)
    print(f"Train: {len_train} Val: {len_val}")
    # Need to make sure batch size is a factor of 2 to execute mixed precision training 
    df_train = df_train[:(CFG.batch_size * (len_train // CFG.batch_size))]

    if CFG.debug:
        df_train = df_train[:100]
        df_val = df_val[:100]
        CFG.epochs = 2 # for debugging


    print(f"Train: {len(df_train)} Val: {len(df_val)}")

    # Setup dataloader
    train_ds = MyDataset(df_train, view)
    val_ds = MyDataset(df_val, view)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
    )

    num_training_steps = len(train_dataloader) * CFG.epochs // CFG.gpus

    hparams = {attr: getattr(CFG, attr) for attr in dir(CFG) if not callable(getattr(CFG, attr)) and not attr.startswith("__")}

    model = Model(model_name = CFG.backbone)

    pl_module = PoseModel(model, hparams, num_training_steps)


    # Define a checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',  # specify what to monitor
        dirpath=f'models_pl/fold_0/{CFG.backbone}_with_16/{view}',  # directory where the model will be saved
        filename='best_checkpoint_{epoch}_{val_acc:.4f}',  # the model filename
        save_top_k=1,  # save only the best model
        mode='max')  # the mode can be either 'min' for metrics like loss, or 'max' for metrics like accuracy
    checkpoint_cb = ModelCheckpoint(CFG.output_path)


    loggers = []

    wandb.finish()
    wandb_logger = WandbLogger(project="PoseTrain", name=f"pose train with {view} 16 dims")

    trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=CFG.epochs,
            val_check_interval=CFG.val_check_interval,
            check_val_every_n_epoch=CFG.check_val_every_n_epoch,
            # gradient_clip_val=CFG.gradient_clip_val,
            precision=16, # if you have tensor cores (t4, v100, a100, etc.) training will be 2x faster
            num_sanity_val_steps=0,
            callbacks=[checkpoint_callback], 
            logger=wandb_logger,
            log_every_n_steps=10, 

    )
    trainer.fit(pl_module, 
                train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader,)


