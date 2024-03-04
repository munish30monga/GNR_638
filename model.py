import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import timm
import torch
from losses import choose_loss_function
    
class FGCM_Model(L.LightningModule):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.cfg = cfg
        self.learning_rate = cfg.learning_rate
        self.save_hyperparameters()  
        self.base_model = timm.create_model(self.cfg.backbone, pretrained=cfg.pretrained)
        self.embedding_size = self.base_model.num_features  
        if 'efficientnet' in self.cfg.backbone and self.cfg.use_spatial_attention:
            self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])
            self.spatial_attention = SpatialAttentionModule(kernel_size=7)
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.projection = nn.Sequential(
                # nn.Dropout(0.25),  # Add dropout before the dense layer as per the new architecture
                nn.Linear(self.embedding_size, num_classes),  # Linear layer
                # nn.Dropout(0.5),   
            )
        else:
            self.base_model = nn.Sequential(*list(self.base_model.children())[:-1]) 
            self.projection = nn.Sequential(
                # nn.Dropout(0.25),  # Add dropout before the dense layer as per the new architecture
                nn.Linear(self.embedding_size, num_classes),           # Linear laye
                # nn.Dropout(0.5),   
            )
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
                            
    def forward(self, x):
        x = self.base_model(x)
        if 'efficientnet' in self.cfg.backbone and self.cfg.use_spatial_attention:
            x = self.spatial_attention(x) * x 
            x = self.pooling(x)  
            x = torch.flatten(x, 1)  
        else:
            x = x.view(x.size()[0], -1)
        x = self.projection(x)
        return x
        
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
            'Adam': torch.optim.Adam(self.parameters(), lr=float(self.learning_rate)),
            'SGD': torch.optim.SGD(self.parameters(), lr=float(self.learning_rate)),
            'AdamW': torch.optim.AdamW(self.parameters(), lr=float(self.learning_rate))
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

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Compute the spatial attention scores
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)