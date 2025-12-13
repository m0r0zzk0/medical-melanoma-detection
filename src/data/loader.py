import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
from .augmentation import get_medical_augmentation
from PIL import Image  

class ISICDataset(Dataset):
    """
    Dataset для загрузки ISIC melanoma images с метаданными.
    
    Args:
        data_dir (str): Путь к папке с изображениями (data/isic)
        metadata_path (str): Путь к metadata.csv
        img_size (int): Размер изображения (по умолчанию 224x224)
    """
    
    def __init__(self, data_dir, metadata_path, img_size=224, train=True):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.train = train
        self.augment = get_medical_augmentation(train=train)
        
        # Загружаем metadata
        self.metadata = pd.read_csv(metadata_path)
        
        # Фильтруем только те, у которых есть diagnosis_1
        self.metadata = self.metadata.dropna(subset=['diagnosis_1'])
        
        # Создаём маппинг диагноза → число
        self.diagnosis_to_label = {
            'Malignant': 1,
            'Benign': 0
        }
        
        print(f"Loaded {len(self.metadata)} images from metadata")
        print(f"Melanoma: {(self.metadata['diagnosis_1'] == 'Malignant').sum()}")
        print(f"Benign: {(self.metadata['diagnosis_1'] == 'Benign').sum()}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        isic_id = row['isic_id']
        diagnosis = row['diagnosis_1']
        
        if diagnosis not in self.diagnosis_to_label:
            return self.__getitem__((idx + 1) % len(self))
        
        img_path = self.data_dir / f"{isic_id}.jpg"
        if not img_path.exists():
            return self.__getitem__((idx + 1) % len(self))
        
        # Читаем как OpenCV
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # ✅ КРИТИЧНО: Конвертируем в PIL для transforms
        img_pil = Image.fromarray(img.astype(np.uint8))
        
        # ✅ Применяем augmentation (работает с PIL)
        img_tensor = self.augment(img_pil)
        
        label = self.diagnosis_to_label[diagnosis]
        
        return {
            'image': img_tensor,
            'label': label,
            'isic_id': isic_id
        }





# ДЛЯ ТЕСТИРОВАНИЯ
if __name__ == "__main__":
    data_dir = "data/isic"
    metadata_path = "data/metadata.csv"
    
    # ✅ Создавай с train=True для augmentation
    dataset = ISICDataset(data_dir, metadata_path, img_size=224, train=True)
    
    print("\nСнимаем несколько примеров:")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"  {i}: {sample['isic_id']} - Label: {sample['label']} - Shape: {sample['image'].shape}")
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    print("\nПроверяем DataLoader:")
    batch = next(iter(dataloader))
    print(f"  Batch size: {batch['image'].shape}")
    print(f"  Labels: {batch['label']}")

