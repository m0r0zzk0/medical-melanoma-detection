import torch
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import (
    confusion_matrix, 
    roc_auc_score, 
    classification_report,
    roc_curve,
    auc
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.loader import ISICDataset
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn

def evaluate():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")
    
    # Load dataset
    dataset = ISICDataset(
        data_dir='data/isic',
        metadata_path='data/metadata.csv',
        img_size=224
    )
    
    val_loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=3)
    
    # Load model
    model = models.resnet50(weights=None)  # ← НЕ загружаем pretrained
    model.fc = nn.Linear(model.fc.in_features, 1)
    
    checkpoint_path = Path('checkpoints/best_model_epoch1.pth')
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print(f"✅ Loaded model from {checkpoint_path}")
    else:
        print(f"❌ Model not found at {checkpoint_path}")
        return
    
    model = model.to(DEVICE)
    model.eval()
    
    # Predictions
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluation"):  # ← PROGRESS BAR!
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Metrics
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(classification_report(
        all_labels, 
        all_preds,
        target_names=['Benign', 'Melanoma'],
        digits=4
    ))
    
    # AUC-ROC
    auc = roc_auc_score(all_labels, all_probs)
    print(f"AUC-ROC Score: {auc:.4f}\n")
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Рассчитай ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    # Найди optimal threshold (Youden Index)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

    print(f"\nOptimal Threshold (Youden): {optimal_threshold:.4f}")
    print(f"  At this threshold:")
    print(f"    Sensitivity (TPR): {tpr[optimal_idx]:.4f}")
    print(f"    Specificity (1-FPR): {1 - fpr[optimal_idx]:.4f}")

    # Применяй этот threshold
    y_pred_optimal = (all_probs >= optimal_threshold).astype(int)

    print(f"\nWith optimal threshold:")
    print(classification_report(all_labels, y_pred_optimal, 
                            target_names=['Benign', 'Melanoma']))

    
    with open('checkpoints/optimal_threshold.txt', 'w') as f:
        f.write(f"{optimal_threshold:.6f}")
        
    print(f"\n Optimal threshold saved to checkpoints/optimal_threshold.txt")
if __name__ == '__main__':
    evaluate()
