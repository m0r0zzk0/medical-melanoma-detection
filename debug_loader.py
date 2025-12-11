import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import ISICDataset

dataset = ISICDataset(
    data_dir='data/isic',
    metadata_path='data/metadata.csv',
    img_size=224
)

# Проверим несколько примеров
print("Первые 10 примеров:")
for i in range(10):
    sample = dataset[i]
    print(f"  Index {i}: label={sample['label']}, image shape={sample['image'].shape}")

# Проверим распределение
import numpy as np
labels = [dataset[i]['label'] for i in range(100)]
print(f"\nLabels в первых 100: {set(labels)}")
print(f"Min: {min(labels)}, Max: {max(labels)}")
