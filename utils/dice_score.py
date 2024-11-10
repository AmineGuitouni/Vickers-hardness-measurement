import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def dice_loss(input: Tensor, target: Tensor):
    # Dice loss (objective to minimize) between 0 and 1
    fn = dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

import numpy as np

def flat_dice_loss(mask1, mask2, smooth=1e-6):
    """
    Compute the Dice loss between two masks.

    Parameters:
    mask1 (numpy array): First mask, usually the predicted mask.
    mask2 (numpy array): Second mask, usually the ground truth mask.
    smooth (float): A small constant to avoid division by zero.

    Returns:
    float: Dice loss between mask1 and mask2.
    """
    # Flatten the masks to ensure the computation is done on the whole array
    mask1_flat = mask1.flatten()
    mask2_flat = mask2.flatten()
    
    # Compute intersection and union
    intersection = np.sum(mask1_flat * mask2_flat)
    union = np.sum(mask1_flat) + np.sum(mask2_flat)
    
    # Compute Dice coefficient
    dice_coeff = (2 * intersection + smooth) / (union + smooth)
    
    # Return Dice loss
    return 1 - dice_coeff