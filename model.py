import lightning as L
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import FeaturePyramidNetwork
import timm
import torch
from losses import choose_loss_function
from collections import OrderedDict
    
class FGCM_Model(L.LightningModule):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg.learning_rate
        self.save_hyperparameters()  
        if cfg.use_fpn:
            self.base_model = timm.create_model(self.cfg.backbone, pretrained=cfg.pretrained, features_only=True, num_classes=num_classes)
            feature_channels = self.base_model.feature_info.channels()
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=feature_channels,
                out_channels=256,
            )
            self.projection = nn.Linear(256 * len(feature_channels), num_classes)
            # self.projection.apply(self.init_weights)
        else:
            self.base_model = timm.create_model(self.cfg.backbone, pretrained=cfg.pretrained, num_classes=num_classes)
            
        self.criterion = choose_loss_function(self.cfg)
        
        # If unfreeze_last_n is -1, make all layers trainable
        if self.cfg.unfreeze_last_n == -1:
            print("=> All layers are trainable.")
            for param in self.base_model.parameters():
                param.requires_grad = True
        else:
            # Freeze all layers initially
            if self.cfg.unfreeze_last_n == 0:
                print("=> All layers are frozen.")
            else:
                print(f"=> Unfreezing the last {self.cfg.unfreeze_last_n} layers.")
            for param in self.base_model.parameters():
                param.requires_grad = False

            # Unfreeze the last n layers
            num_layers = len(list(self.base_model.children()))
            for i, child in enumerate(self.base_model.children()):
                if i >= num_layers - self.cfg.unfreeze_last_n:
                    for param in child.parameters():
                        param.requires_grad = True
    
    def init_weights(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_normal_(layer.weight)   
                        
    def forward(self, x):
        if self.cfg.use_fpn:
            features = self.base_model(x)
            
            # Create an OrderedDict for FPN input
            fpn_input = OrderedDict([
                (f'feat{i}', feature) for i, feature in enumerate(features)
            ])
            
            # FPN Output
            fpn_output = self.fpn(fpn_input)
            
            combined_features = torch.cat([torch.nn.functional.adaptive_avg_pool2d(output, (1, 1)) for output in fpn_output.values()], dim=1)
            combined_features = combined_features.view(combined_features.size(0), -1)
            
            # Final classification
            logits = self.projection(combined_features)
        else:
            x = self.base_model(x)
            logits = x / self.cfg.temperature
        return logits
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        train_loss = self.criterion(F.softmax(logits, dim=1), y) if self.cfg.loss_function == 'FocalLoss' else self.criterion(logits, y)
        self.log('train_loss', train_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(F.softmax(logits, dim=1), y) if self.cfg.loss_function == 'FocalLoss' else self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.tensor(torch.sum(preds == y).item() / len(preds), device=self.device)*100
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'test_acc': acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(F.softmax(logits, dim=1), y) if self.cfg.loss_function == 'FocalLoss' else self.criterion(logits, y) 
        preds = torch.argmax(logits, dim=1)
        acc = torch.tensor(torch.sum(preds == y).item() / len(preds), device=self.device)*100
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):        
        optimizer = {
            'Adam': torch.optim.Adam(self.parameters(), lr=float(self.learning_rate)),
            'SGD': torch.optim.SGD(self.parameters(), lr=float(self.learning_rate)),
            'AdamW': torch.optim.AdamW(self.parameters(), lr=float(self.learning_rate), weight_decay=float(self.cfg.weight_decay)),
        }[self.cfg.optimizer]
        print(f"=> Using '{self.cfg.optimizer}' optimizer.")
        
        scheduler = {
            'CosineAnnealing': {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, eta_min=0),
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
