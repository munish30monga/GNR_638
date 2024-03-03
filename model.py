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

class FGCM_Model(pl.LightningModule):
    def __init__(self, base_model_name, num_classes, optimizer_type, learning_rate=1e-3, weight_decay=0.0):
        super().__init__()
        self.save_hyperparameters()  
        self.base_model = timm.create_model(base_model_name, pretrained=True)
        self.embedding_size = self.base_model.num_features  # Get the number of features (embedding size) from the base model
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) # Remove the classification head
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        
        self.projection = nn.Sequential(
            nn.Linear(self.embedding_size, num_classes),            # Linear layer
            nn.BatchNorm1d(num_classes)) 

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size()[0], -1)
        x = self.projection(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.tensor(torch.sum(preds == y).item() / len(preds), device=self.device)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.tensor(torch.sum(preds == y).item() / len(preds), device=self.device)
        self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', acc, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        optimizer = {
            'Adam': torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.hparams.weight_decay),
            'SGD': torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.hparams.weight_decay)
        }[self.hparams.optimizer_type]
        return optimizer
    