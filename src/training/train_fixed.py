import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
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
    
    # ✅ ИСПРАВКА: Создай ДВА отдельных dataset
    # Dataset с augmentation для training
    train_dataset_full = ISICDataset(
        data_dir='data/isic',
        metadata_path='data/metadata.csv',
        img_size=IMG_SIZE,
        train=True  # ✅ С augmentation
    )
    
    # Dataset БЕЗ augmentation для валидации
    val_dataset_full = ISICDataset(
        data_dir='data/isic',
        metadata_path='data/metadata.csv',
        img_size=IMG_SIZE,
        train=False  # ✅ БЕЗ augmentation
    )
    
    # Split на train/val (80/20)
    train_size = int(0.8 * len(train_dataset_full))
    val_size = int(0.2 * len(val_dataset_full))
    
    train_dataset, _ = random_split(
        train_dataset_full, 
        [train_size, len(train_dataset_full) - train_size]
    )
    _, val_dataset = random_split(
        val_dataset_full,
        [len(val_dataset_full) - val_size, val_size]
    )
    
    print(f"Train samples: {len(train_dataset)} (with augmentation)")
    print(f"Val samples: {len(val_dataset)} (no augmentation)")
    
    # ============== OVERSAMPLING ==============
    # Найди melanoma samples в training set
    train_melanoma_count = 0
    train_benign_count = 0
    
    for idx in train_dataset.indices:
        diagnosis = train_dataset_full.metadata.iloc[idx]['diagnosis_1']
        if diagnosis == 'Malignant':
            train_melanoma_count += 1
        else:
            train_benign_count += 1
    
    print(f"\nTrain set composition:")
    print(f"  Melanoma: {train_melanoma_count}")
    print(f"  Benign: {train_benign_count}")
    
    # Oversample melanoma (дублируй для баланса)
    oversampling_factor = int(train_benign_count / train_melanoma_count)
    print(f"  Oversampling factor: {oversampling_factor}x")
    
    # Создай список всех индексов с oversampling
    train_indices = list(train_dataset.indices)
    melanoma_indices_in_train = [
        i for i in train_indices 
        if train_dataset_full.metadata.iloc[i]['diagnosis_1'] == 'Malignant'
    ]
    
    # Добавь дополнительные копии melanoma samples
    oversampled_indices = train_indices + melanoma_indices_in_train * (oversampling_factor - 1)
    print(f"  Total after oversampling: {len(oversampled_indices)}")
    
    # Создай новый dataset с oversampling
    oversampled_dataset = Subset(train_dataset_full, oversampled_indices)
    
    # ============== DATALOADER ==============
    train_loader = DataLoader(
        oversampled_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=3,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=3,
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
    
    # ============== RESUME TRAINING ✅ ==============
    checkpoint_dir = Path('checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    
    start_epoch = 0
    best_val_acc = 0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Проверь есть ли checkpoint для продолжения
    last_checkpoints = sorted(checkpoint_dir.glob('last_model_epoch*.pth'))
    if last_checkpoints:
        latest = last_checkpoints[-1]
        print(f"\n✅ Found checkpoint: {latest}")
        model.load_state_dict(torch.load(latest, map_location=DEVICE))
        
        # Извлеки номер эпохи
        epoch_num = int(latest.stem.split('epoch')[-1])
        start_epoch = epoch_num
        print(f"📍 Resuming from epoch {epoch_num}, will train epochs {epoch_num + 1} to {NUM_EPOCHS}")
        
        # Загрузи history если есть
        history_path = checkpoint_dir / 'training_history.json'
        if history_path.exists():
            with open(history_path, 'r') as f:
                loaded_history = json.load(f)
                training_history['train_loss'] = loaded_history.get('train_loss', [])
                training_history['train_acc'] = loaded_history.get('train_acc', [])
                training_history['val_loss'] = loaded_history.get('val_loss', [])
                training_history['val_acc'] = loaded_history.get('val_acc', [])
                best_val_acc = max(training_history['val_acc']) if training_history['val_acc'] else 0
    else:
        print("\n🆕 No checkpoint found, starting fresh training")
    
    # ============== TRAINING ==============
    print(f"\nStarting training from epoch {start_epoch+1} to {NUM_EPOCHS}...\n")
    
    for epoch in range(start_epoch, NUM_EPOCHS):
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
        
        # ✅ Сохрани как последнюю (для resume)
        last_checkpoint_path = checkpoint_dir / f'last_model_epoch{epoch+1}.pth'
        torch.save(model.state_dict(), last_checkpoint_path)
        print(f"  💾 Saved checkpoint: {last_checkpoint_path}")
        
        # Сохрани history после каждой эпохи
        history_path = checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
    
    # ============== FINAL RESULTS ==============
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Save final model
    final_model_path = checkpoint_dir / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model: {final_model_path}")
    
    print(f"\n📊 Training History:")
    for i, (tl, ta, vl, va) in enumerate(zip(
        training_history['train_loss'],
        training_history['train_acc'],
        training_history['val_loss'],
        training_history['val_acc']
    )):
        print(f"  Epoch {i+1}: Train Loss={tl:.4f}, Train Acc={ta:.4f}, Val Loss={vl:.4f}, Val Acc={va:.4f}")



if __name__ == '__main__':
    main()
