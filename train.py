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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

def train_model(cfg , model, data_module, max_epochs, accelerator, devices, project_name):

    # Logger
    wandb_logger = WandbLogger(project='GNR_638', log_model='all', name=f'{project_name}')
        
    # Model checkpoint callback to save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='test_loss',
        filename='model-{epoch:02d}-{test_loss:.2f}',
        save_top_k=1,
        mode='min',
    )
    
    # Initialize a trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        accelerator=accelerator,
        devices=devices,
    )
    
    # Train the model
    trainer.fit(model, train_dataloader=data_module.train_dataloader(), val_dataloaders=data_module.test_dataloader())
    
    return trainer, model