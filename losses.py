import torch
import torch.nn as nn
import torch.nn.functional as F
from focal_loss.focal_loss import FocalLoss

        
def choose_loss_function(cfg):
    if cfg.label_smoothing and cfg.loss_function == 'CrossEntropy':
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    
    if cfg.loss_function == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    
    if cfg.loss_function == 'FocalLoss':
        return FocalLoss(gamma=cfg.gamma)
