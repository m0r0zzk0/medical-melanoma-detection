import torch
import torchvision.transforms as transforms
import numpy as np

def get_medical_augmentation(train=True):
    """
    Medical-safe augmentations используя встроенные PyTorch transforms.
    Не требует albumentations - только стандартная torchvision!
    """
    if train:
        return transforms.Compose([
            # Rotation ±15 degrees (medical-safe)
            transforms.RandomRotation(degrees=15),
            
            # Horizontal flip (safe for dermoscopy)
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Random brightness/contrast
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            
        
            transforms.ToTensor(),  # PIL Image → Tensor
            
            # Normalization (ImageNet standards)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        # Validation - only normalization
        return transforms.Compose([
            transforms.ToTensor(),  # ✅ ДОБАВИЛИ
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
