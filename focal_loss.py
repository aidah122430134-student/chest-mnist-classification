# focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss untuk binary classification.
    Focal loss mengurangi weight untuk easy examples dan fokus pada hard examples.
    Sangat efektif untuk medical imaging dengan class imbalance.
    
    Formula: FL = -alpha * (1-pt)^gamma * log(pt)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits dari model (before sigmoid), shape [batch_size, 1]
            targets: ground truth labels, shape [batch_size, 1]
        """
        # Apply sigmoid untuk mendapat probabilities
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = p jika y=1, 1-p jika y=0
        
        # Focal loss computation
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Weighted BCE Loss untuk handling class imbalance.
    Memberikan weight lebih besar pada minority class.
    """
    def __init__(self, pos_weight=None):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        if self.pos_weight is not None:
            return F.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=self.pos_weight
            )
        else:
            return F.binary_cross_entropy_with_logits(inputs, targets)
