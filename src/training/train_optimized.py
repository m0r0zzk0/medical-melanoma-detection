import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, ConcatDataset, Subset
from torchvision import models
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import json
from datetime import datetime


# Добавляем путь к src для импортов
sys.path.insert(0, str(Path(__file__).parent.parent))


from data.loader import ISICDataset



class MetricTracker:
    """Отслеживает метрики обучения"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val += val * n
        self.count += n
    
    def avg(self):
        return self.val / self.count if self.count != 0 else 0



def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Один epoch обучения"""
    model.train()
    losses = MetricTracker()
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device).float().unsqueeze(1)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()

        
        # Метрики
        losses.update(loss.item(), images.size(0))
        
        # Accuracy
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels.long()).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    return losses.avg(), correct / total



def validate(model, val_loader, criterion, device):
    """Валидация"""
    model.eval()
    losses = MetricTracker()
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc="Validation")
    
    with torch.no_grad():
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            losses.update(loss.item(), images.size(0))
            
            # Accuracy
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{losses.avg():.4f}',
                'acc': f'{correct/total:.4f}'
            })
    
    return losses.avg(), correct / total



def main():
    # ============== CONFIG ==============
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 3
    IMG_SIZE = 224
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # ============== DATASET & DATALOADER ==============
    print("\nLoading dataset...")
    
    # ✅ Создаём dataset с train=True для augmentation
    dataset = ISICDataset(
        data_dir='data/isic',
        metadata_path='data/metadata.csv',
        img_size=IMG_SIZE,
        train=True
    )
    
    # Split на train/val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {train_size}")
    print(f"Val samples: {val_size}")
    
    # ============== OVERSAMPLING ✅ ДОБАВИЛИ ==============
    # Найди melanoma samples в training set
    train_melanoma_count = 0
    train_benign_count = 0
    
    for idx in train_dataset.indices:
        diagnosis = dataset.metadata.iloc[idx]['diagnosis_1']
        if diagnosis == 'Malignant':
            train_melanoma_count += 1
        else:
            train_benign_count += 1
    
    print(f"\nTrain set composition:")
    print(f"  Melanoma: {train_melanoma_count}")
    print(f"  Benign: {train_benign_count}")
    
    # Oversample melanoma (дублируй ~30 раз для баланса)
    oversampling_factor = int(train_benign_count / train_melanoma_count)
    print(f"  Oversampling factor: {oversampling_factor}x")
    
    # Создай список всех индексов с oversampling
    train_indices = list(train_dataset.indices)
    melanoma_indices_in_train = [
        i for i in train_indices 
        if dataset.metadata.iloc[i]['diagnosis_1'] == 'Malignant'
    ]
    
    # Добавь дополнительные копии melanoma samples
    oversampled_indices = train_indices + melanoma_indices_in_train * (oversampling_factor - 1)
    print(f"  Total after oversampling: {len(oversampled_indices)}")
    
    # Создай новый dataset с oversampling
    oversampled_dataset = Subset(dataset, oversampled_indices)
    
    # ============== DATALOADER ==============
    train_loader = DataLoader(
        oversampled_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=3,  # ✅ УВЕЛИЧИЛИ с 0 на 3
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,  # ✅ ДОБАВИЛИ
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # ============== MODEL ==============
    print("\nLoading ResNet50...")
    model = models.resnet50(pretrained=True)
    
    # Заменяем последний слой (2048 → 1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1)  # Binary classification
    
    model = model.to(DEVICE)
    
    # Считаем параметры
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # ============== LOSS & OPTIMIZER ==============
    # Вычисляем веса классов
    class_counts = np.array([32539, 584])  # benign, melanoma
    total = class_counts.sum()
    class_weights = total / (2 * class_counts)
    
    print(f"Class weights: benign={class_weights[0]:.2f}, melanoma={class_weights[1]:.2f}")
    
    # Используем pos_weight для melanoma (класс 1)
    pos_weight = class_weights[1] / class_weights[0]
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=DEVICE))
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ============== TRAINING ==============
    print(f"\nStarting training for {NUM_EPOCHS} epochs...\n")
    
    best_val_acc = 0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch+1
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # History
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = checkpoint_dir / f'best_model_epoch{epoch+1}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✅ Saved best model: {checkpoint_path}")
    
    # ============== FINAL RESULTS ==============
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Save training history
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Saved training history: {history_path}")
    
    # Save final model
    final_model_path = checkpoint_dir / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")



if __name__ == '__main__':
    main()
