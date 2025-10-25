"""
Evaluation functions for OOB Detection
Handles cross-validation and metrics calculation
"""

import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from dataset import OOBDatasetWithColor
from train import train_fold
from utils import set_seed, create_fold_split
from config import N_FOLDS, BATCH_SIZE, EPOCHS, RANDOM_SEED


def cross_validate(video_paths, labels, jersey_colors):
    """
    Perform k-fold cross-validation
    
    Args:
        video_paths: List of video file paths
        labels: List of labels
        jersey_colors: List of jersey colors
        
    Returns:
        fold_results: List of validation accuracies per fold
        all_predictions: List of all predictions across folds
    """
    if len(video_paths) != 20:
        print(f"⚠️ Expected 20 videos, found {len(video_paths)}")
    
    fold_results = []
    all_predictions = []
    
    print(f"\n🔀 Creating {N_FOLDS} folds with {len(video_paths) // N_FOLDS} videos each")
    
    for fold in range(N_FOLDS):
        # Set seed for this fold
        set_seed(RANDOM_SEED + fold)
        
        # Create train/val split
        train_paths, train_labels, train_colors, val_paths, val_labels, val_colors = \
            create_fold_split(video_paths, labels, jersey_colors, fold, N_FOLDS)
        
        print(f"\nFold {fold+1}: Train on {len(train_paths)} videos, Validate on {len(val_paths)} videos")
        print(f"Val videos: {[Path(p).name[:15] for p in val_paths]}")
        
        # Create datasets
        train_dataset = OOBDatasetWithColor(
            train_paths, train_labels, train_colors,
            is_train=True, augment=True
        )
        val_dataset = OOBDatasetWithColor(
            val_paths, val_labels, val_colors,
            is_train=False, augment=False
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        # Train this fold
        val_acc, predictions = train_fold(train_loader, val_loader, fold+1, epochs=EPOCHS)
        
        fold_results.append(val_acc)
        all_predictions.extend(predictions)
    
    return fold_results, all_predictions


def print_results(fold_results, all_predictions):
    """
    Print comprehensive results
    
    Args:
        fold_results: List of validation accuracies per fold
        all_predictions: List of all predictions
    """
    print(f"\n{'='*60}")
    print("📊 CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    
    for i, acc in enumerate(fold_results):
        print(f"Fold {i+1}: {acc:.1f}%")
    
    mean_acc = np.mean(fold_results)
    std_acc = np.std(fold_results)
    
    print(f"\n🎯 Mean Accuracy: {mean_acc:.1f}% ± {std_acc:.1f}%")
    
    # Overall accuracy
    overall_correct = sum(p['correct'] for p in all_predictions)
    overall_acc = 100 * overall_correct / len(all_predictions)
    print(f"📈 Overall Accuracy: {overall_acc:.1f}% ({overall_correct}/{len(all_predictions)})")
    
    # Performance by label
    home_preds = [p for p in all_predictions if p['true'] == 1]
    away_preds = [p for p in all_predictions if p['true'] == 0]
    
    home_acc = 100 * sum(p['correct'] for p in home_preds) / len(home_preds) if home_preds else 0
    away_acc = 100 * sum(p['correct'] for p in away_preds) / len(away_preds) if away_preds else 0
    
    print(f"\n📊 Performance by Label:")
    print(f"  Home team ball: {home_acc:.1f}% ({sum(p['correct'] for p in home_preds)}/{len(home_preds)})")
    print(f"  Away team ball: {away_acc:.1f}% ({sum(p['correct'] for p in away_preds)}/{len(away_preds)})")
    
    print("\n✅ Cross-validation complete!")
