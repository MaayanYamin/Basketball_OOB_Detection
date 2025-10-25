"""
Main entry point for OOB Detection
Runs cross-validation and prints results
"""

import warnings
from pathlib import Path

from utils import set_seed, prepare_data
from evaluate import cross_validate, print_results
from config import RANDOM_SEED, VIDEO_DIR

warnings.filterwarnings('ignore')


def main():
    """Main function to run cross-validation"""
    
    # Set random seed for reproducibility
    set_seed(RANDOM_SEED)
    
    print("🏀 OOB Detection - 4-Fold Cross Validation")
    print("="*60)
    print(f"Random Seed: {RANDOM_SEED}")
    print()
    
    # Check if videos are available
    video_count = len(list(Path(VIDEO_DIR).glob('clip*.mp4')))
    if video_count < 20:
        print("⚠️ Please upload all 20 video clips first!")
        print(f"Found only {video_count} videos in {VIDEO_DIR}")
        print("Use the Files sidebar to upload your clips to /content/")
        return
    
    # Prepare data
    video_paths, labels, jersey_colors = prepare_data()
    
    # Run cross-validation
    fold_results, all_predictions = cross_validate(video_paths, labels, jersey_colors)
    
    # Print comprehensive results
    print_results(fold_results, all_predictions)


if __name__ == "__main__":
    main()
