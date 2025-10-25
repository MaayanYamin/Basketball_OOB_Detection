"""
Training functions for OOB Detection
Handles single fold training loop
"""

import torch
import torch.nn as nn

from model import OOBModelWithColor
from config import LEARNING_RATE, LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR


def train_fold(train_loader, val_loader, fold_num, epochs=5):
    """
    Train one fold
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        fold_num: Fold number (for logging)
        epochs: Number of epochs to train
        
    Returns:
        best_val_acc: Best validation accuracy achieved
        best_predictions: Predictions from best epoch
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = OOBModelWithColor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=LR_SCHEDULER_PATIENCE, 
        factor=LR_SCHEDULER_FACTOR
    )
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_predictions = []
    
    print(f"\n{'-'*60}")
    print(f"Training Fold {fold_num}")
    print(f"{'-'*60}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            frames = batch['frames'].to(device)
            labels = batch['label'].to(device)
            colors = batch['color'].to(device)
            
            optimizer.zero_grad()
            outputs = model(frames, colors)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_predictions = []
        
        with torch.no_grad():
            for batch in val_loader:
                frames = batch['frames'].to(device)
                labels = batch['label'].to(device)
                colors = batch['color'].to(device)
                
                outputs = model(frames, colors)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                for i in range(len(predicted)):
                    val_predictions.append({
                        'video': batch['video_name'][i],
                        'pred': predicted[i].item(),
                        'true': labels[i].item(),
                        'correct': predicted[i].item() == labels[i].item()
                    })
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_loss = train_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_loss:.3f}, "
              f"Train Acc={train_acc:.1f}%, Val Acc={val_acc:.1f}%")
        
        scheduler.step(avg_loss)
        
        # Save best predictions
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_predictions = val_predictions.copy()
    
    # Print best predictions
    print(f"\nFold {fold_num} Best Predictions (Best Val Acc: {best_val_acc:.1f}%):")
    for pred in best_predictions:
        status = "✓" if pred['correct'] else "✗"
        pred_label = "Home" if pred['pred'] == 1 else "Away"
        true_label = "Home" if pred['true'] == 1 else "Away"
        print(f"  {status} {pred['video'][:30]:30} | Pred: {pred_label:4} | True: {true_label:4}")
    
    return best_val_acc, best_predictions
