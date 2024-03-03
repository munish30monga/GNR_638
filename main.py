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
import argparse
import yaml
import munch
from train import train_model
from dataloader import CUB_DataModule, dataset_summary
from model import FGCM_Model

project_name = 'test_run'
# wandb.init(name=f'{project_name}')
wandb.init(mode='dryrun')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fine Grained Classification Model.")
    parser.add_argument(
        "--config_file",
        default="configs/aio.yaml",
        type=str,
        help="path to config file.",
    )
    args = parser.parse_args()

    # Load config yaml file as nested object
    cfg = yaml.safe_load(open(args.config_file, "r"))
    cfg = munch.munchify(cfg)
    wandb.config.update(cfg)
    wandb.save(f'./configs/aio.yaml')
    
    dataset_summary_dict = dataset_summary(cfg.dataset_dir)
    data_module = CUB_DataModule(cfg.dataset_dir, cfg.batch_size, cfg.num_workers)
    data_module.setup()
    num_classes = dataset_summary_dict['num_classes']
    
    base_model = 'resnet18'
    model = FGCM_Model(base_model, num_classes, cfg.optimizer_type, cfg.learning_rate, cfg.weight_decay)
    print(f"Fine-Grained Classification Model is build using '{base_model}' as base model")
    
    train_model(cfg, model, data_module, max_epochs=5, accelerator='gpu', devices=1, project_name=project_name)
    
    