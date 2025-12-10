import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch

class ISICDataset(Dataset):
    """
    Dataset для загрузки ISIC melanoma images с метаданными.
    
    Args:
        data_dir (str): Путь к папке с изображениями (data/isic)
        metadata_path (str): Путь к metadata.csv
        img_size (int): Размер изображения (по умолчанию 224x224)
    """
    
    def __init__(self, data_dir, metadata_path, img_size=224, transform=None):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.transform = transform
        
        # Загружаем metadata
        self.metadata = pd.read_csv(metadata_path)
        
        # Фильтруем только те, у которых есть diagnosis_1
        self.metadata = self.metadata.dropna(subset=['diagnosis_1'])
        
        # Создаём маппинг диагноза → число
        self.diagnosis_to_label = {
            'Melanoma': 1,
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
        
        # Путь к изображению
        img_path = self.data_dir / f"{isic_id}.jpg"
        
        # Загружаем изображение
        if not img_path.exists():
            print(f"Warning: {img_path} not found")
            return None
        
        img = cv2.imread(str(img_path))
        
        # Конвертируем BGR в RGB (OpenCV загружает в BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Ресайзим до нужного размера
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Нормализуем (значения 0-1)
        img = img.astype(np.float32) / 255.0
        
        # Конвертируем диагноз в число
        label = self.diagnosis_to_label.get(diagnosis, -1)
        
        # Если нужны аугментации (потом добавим)
        if self.transform:
            img = self.transform(img)
        
        # Конвертируем в torch tensor
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        
        return {
            'image': img,
            'label': label,
            'isic_id': isic_id
        }


# ДЛЯ ТЕСТИРОВАНИЯ
if __name__ == "__main__":
    # Пути
    data_dir = "data/isic"
    metadata_path = "data/metadata.csv"
    
    # Создаём dataset
    dataset = ISICDataset(data_dir, metadata_path, img_size=224)
    
    # Проверяем несколько примеров
    print("\nСнимаем несколько примеров:")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"  {i}: {sample['isic_id']} - Label: {sample['label']} - Shape: {sample['image'].shape}")
    
    # Создаём DataLoader (для batching)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
    
    print("\nПроверяем DataLoader:")
    batch = next(iter(dataloader))
    print(f"  Batch size: {batch['image'].shape}")
    print(f"  Labels: {batch['label']}")
