import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision.datasets.utils import download_url
from pathlib import Path
import numpy as np
from prettytable import PrettyTable
import torch
import torch.nn as nn
import timm
import random
from thop import clever_format
from torchsummary import summary
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb
from model import FGCM_Model

torch.set_float32_matmul_precision('medium') 

def train_model(cfg , model, data_module, max_epochs, accelerator, devices, logger):  
    # Callbacks      
    checkpoint_callback = ModelCheckpoint(
        dirpath='./checkpoints',
        monitor='val_acc',
        filename='{cfg.backbone}_{epoch:02d}_{acc:.2f}',
        save_top_k=1,
        mode='max',
        verbose=True,
    )
    LR_monitor = LearningRateMonitor(logging_interval='step', log_weight_decay=True)
    
    # Initialize a trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=1,
        callbacks = [checkpoint_callback, LR_monitor],
        logger=logger,
        accelerator=accelerator,
        devices=devices,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)
    
    # Load the best model saved by the checkpoint callback
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")
    if best_model_path:
        best_model = FGCM_Model.load_from_checkpoint(best_model_path)
    
    # Run the test using the best model
    trainer.test(best_model, datamodule=data_module)
    
    return trainer, best_model