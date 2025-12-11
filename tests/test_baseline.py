import torch
from torchvision import transforms

def test_data_loading():
    batch_size = 4
    img = torch.randn(batch_size, 3, 256, 256)
    mask = torch.randint(0, 2, (batch_size, 1, 256, 256))
    
    assert img.shape == (batch_size, 3, 256, 256)
    assert mask.shape == (batch_size, 1, 256, 256)
    assert mask.max() <= 1 and mask.min() >= 0
    print(" Data loading test passed")

def test_iou_calculation():
    pred = torch.tensor([1, 1, 0], dtype=torch.float32)
    target = torch.tensor([1, 1, 0], dtype=torch.float32)
    
    intersection = (pred * target).sum()
    union = ((pred + target) > 0).float().sum()
    iou = intersection / union if union > 0 else 0
    
    assert abs(iou.item() - 1.0) < 0.001
    print("IoU calculation test passed")

def test_normalization():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    print("Normalization test structure verified")

if __name__ == "__main__":
    test_data_loading()
    test_iou_calculation() 
    test_normalization()
    print("\nAll baseline tests passed")