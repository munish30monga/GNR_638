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
import torch.nn.functional as F
import timm
import random
from thop import clever_format
from torchsummary import summary
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from losses import choose_loss_function
    
class FGCM_Model(pl.LightningModule):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.save_hyperparameters()  
        self.base_model = timm.create_model(cfg.backbone, pretrained=True)
        self.embedding_size = self.base_model.num_features  
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) 
        self.cfg = cfg
        self.criterion = choose_loss_function(cfg)
        
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_size, num_classes),           # Linear layer
        ) 
        
        # If unfreeze_last_n is -1, make all layers trainable
        if cfg.unfreeze_last_n == -1:
            print("=> All layers are trainable.")
            for param in self.base_model.parameters():
                param.requires_grad = True
        else:
            # Freeze all layers initially
            if cfg.unfreeze_last_n == 0:
                print("=> All layers are frozen.")
            else:
                print(f"=> Unfreezing the last {cfg.unfreeze_last_n} layers.")
            for param in self.base_model.parameters():
                param.requires_grad = False

            # Unfreeze the last n layers
            num_layers = len(list(self.base_model.children()))
            for i, child in enumerate(self.base_model.children()):
                if i >= num_layers - cfg.unfreeze_last_n:
                    for param in child.parameters():
                        param.requires_grad = True
                        
    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size()[0], -1)
        x = self.projection(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        train_loss = self.criterion(F.softmax(logits, dim=1), y) if self.cfg.loss_function == 'FocalLoss' else self.criterion(logits, y)
        self.log('train_loss', train_loss, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(F.softmax(logits, dim=1), y) if self.cfg.loss_function == 'FocalLoss' else self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.tensor(torch.sum(preds == y).item() / len(preds), device=self.device)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'test_acc': acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(F.softmax(logits, dim=1), y) if self.cfg.loss_function == 'FocalLoss' else self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.tensor(torch.sum(preds == y).item() / len(preds), device=self.device)
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):        
        optimizer = {
            'Adam': torch.optim.Adam(self.parameters(), lr=float(self.cfg.learning_rate), weight_decay=float(self.cfg.weight_decay)),
            'SGD': torch.optim.SGD(self.parameters(), lr=float(self.cfg.learning_rate), weight_decay=float(self.cfg.weight_decay)),
            'AdamW': torch.optim.AdamW(self.parameters(), lr=float(self.cfg.learning_rate), weight_decay=float(self.cfg.weight_decay))
        }[self.cfg.optimizer]
        print(f"=> Using '{self.cfg.optimizer}' optimizer.")
        
        scheduler = {
            'CosineAnnealing': {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=0),
                'interval': 'epoch',
                'frequency': 1
            },
            'ReduceLROnPlateau': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.cfg.patience, min_lr=0, factor=self.cfg.decay_factor),
                'monitor': 'val_loss',  
                'interval': 'epoch',
                'frequency': 1
            }
        }[self.cfg.scheduler]
        print(f"=> Using '{self.cfg.scheduler}' scheduler.")
        
        return [optimizer], [scheduler] 
    