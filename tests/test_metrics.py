import torch
import numpy as np

def test_dice_score():
    pred = torch.tensor([1, 1, 0, 0], dtype=torch.float32)
    target = torch.tensor([1, 0, 1, 0], dtype=torch.float32)
    
    intersection = (pred * target).sum()
    dice = (2 * intersection) / (pred.sum() + target.sum())
    
    expected = 0.5 
    assert abs(dice.item() - expected) < 0.001
    print("Dice score test passed")

def test_mask_processing():
    original_mask = torch.tensor([1, 2, 3, 1, 2], dtype=torch.long)
    binary_mask = (original_mask == 1).long()
    
    assert torch.all(binary_mask == torch.tensor([1, 0, 0, 1, 0]))
    print("Mask processing test passed")

if __name__ == "__main__":
    test_dice_score()
    test_mask_processing()
    print("\nMetrics tests complete")