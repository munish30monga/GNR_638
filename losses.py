import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
def choose_loss_function(cfg):
    if cfg.loss_function == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    elif cfg.label_smoothing:
        return nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    elif cfg.loss_function == 'FocalLoss':
        return FocalLoss(alpha=cfg.alpha, gamma=cfg.gamma, reduction='mean')