"""
Utility functions for OOB Detection
Includes data loading, seeding, and helper functions
"""

import random
import numpy as np
import torch
from pathlib import Path

from config import JERSEY_COLORS, VIDEO_DIR, VIDEO_PATTERN, RANDOM_SEED


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data():
    """
    Load all videos and prepare for cross-validation
    
    Returns:
        video_paths: List of video file paths
        labels: List of labels (0=away, 1=home)
        jersey_colors: List of jersey colors for each video
    """
    video_dir = Path(VIDEO_DIR)
    
    video_paths = []
    labels = []
    jersey_colors = []
    
    for video_file in sorted(video_dir.glob(VIDEO_PATTERN)):
        filename = video_file.stem
        clip_num = filename.split(' - ')[0].lower()
        
        # Get label
        if "away" in filename.lower():
            label = 0  # Away = 0
        elif "home" in filename.lower():
            label = 1  # Home = 1
        else:
            continue
        
        # Get jersey color
        jersey_color = JERSEY_COLORS.get(clip_num, 'blue')
        
        video_paths.append(str(video_file))
        labels.append(label)
        jersey_colors.append(jersey_color)
    
    print(f"Found {len(video_paths)} videos")
    print(f"  Home (1): {sum(labels)}, Away (0): {len(labels) - sum(labels)}")
    print(f"  White jerseys: {jersey_colors.count('white')}")
    print(f"  Blue jerseys: {jersey_colors.count('blue')}")
    print(f"  Black jerseys: {jersey_colors.count('black')}")
    
    return video_paths, labels, jersey_colors


def create_fold_split(video_paths, labels, jersey_colors, fold, n_folds):
    """
    Create train/val split for a specific fold
    
    Args:
        video_paths: List of all video paths
        labels: List of all labels
        jersey_colors: List of all jersey colors
        fold: Current fold number (0-indexed)
        n_folds: Total number of folds
        
    Returns:
        train_paths, train_labels, train_colors, val_paths, val_labels, val_colors
    """
    indices = list(range(len(video_paths)))
    random.seed(RANDOM_SEED)
    random.shuffle(indices)
    
    fold_size = len(video_paths) // n_folds
    val_start = fold * fold_size
    val_end = val_start + fold_size
    val_indices = indices[val_start:val_end]
    train_indices = [i for i in indices if i not in val_indices]
    
    train_paths = [video_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    train_colors = [jersey_colors[i] for i in train_indices]
    
    val_paths = [video_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    val_colors = [jersey_colors[i] for i in val_indices]
    
    return train_paths, train_labels, train_colors, val_paths, val_labels, val_colors
