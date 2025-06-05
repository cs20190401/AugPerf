import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
  def __init__(self, gamma=2, alpha=0.75, epsilon=1e-7, reduce=True):
    super().__init__()
    self.gamma = gamma
    self.alpha = alpha
    self.epsilon = epsilon
    self.reduce = reduce

  def forward(self, p, labels, mask=None, weight=None):
    # labels: [Batch] (binary labels: 0 or 1)
    # p: [Batch] (probabilities for the positive class, should be after sigmoid)
    
    # Predicted probabilities for the negative class
    q = 1 - p  # [Batch]
    
    # Cast labels to float
    labels_pos = labels.float()  # [Batch]
    labels_neg = 1 - labels_pos  # [Batch]
    
    # Loss for the positive examples
    loss_pos = -self.alpha * (q ** self.gamma) * torch.log(p + self.epsilon)  # [Batch]
    
    # Loss for the negative examples
    loss_neg = -(1 - self.alpha) * (p ** self.gamma) * torch.log(q + self.epsilon)  # [Batch]
    
    # Combine loss terms
    loss = labels_pos * loss_pos + labels_neg * loss_neg  # [Batch]
    
    # Apply optional weight
    if weight is not None:
      loss *= weight  # [Batch]
    
    # Apply optional mask
    if mask is not None:
      loss *= mask  # [Batch]

    loss = torch.sum(loss, dim=1)
    
    # Reduce the loss if required
    return loss.mean() if self.reduce else loss

class DiceLoss(nn.Module):
  def __init__(self, epsilon=1e-5, reduce=True):
    super().__init__()
    self.epsilon = epsilon
    self.reduce = reduce

  def forward(self, p, labels, mask=None):
    # labels: [Batch] (binary labels: 0 or 1)
    # p: [Batch] (probabilities for the positive class, should be after sigmoid)
    
    # Apply optional mask if provided
    if mask is not None:
      p = p * mask
      labels = labels * mask
    
    # Dice Loss calculation
    numerator = 2 * torch.sum(p * labels) + self.epsilon  # [Batch]
    denominator = torch.sum(p ** 2) + torch.sum(labels ** 2) + self.epsilon  # [Batch]
    dice_coef = numerator / denominator  # [Batch]
    
    # Dice loss is the negative log of the Dice coefficient
    loss = -torch.log(dice_coef)  # [Batch]
    
    # Return the mean loss across the batch
    return loss.mean() if self.reduce else loss