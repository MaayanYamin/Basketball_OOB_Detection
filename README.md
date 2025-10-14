# Basketball Out-of-Bounds Detection
Deep learning system to determine which team gets possession after an out-of-bounds call.

## Problem
Out-of-bounds calls often require referees to review video footage at courtside monitors, causing game delays that disrupt momentum and allow players to cool down. This system provides instant automated decisions to eliminate review delays.

## Approach
- **Architecture**: MobileNetV2 + LSTM
- **Transfer Learning**: Leveraged ImageNet pretraining
- **Dataset**: 20 real game clips from Golden State Warriors games challenges (courtesy to the official NBA website)

## Results
- **85% accuracy** across 4 folds (4-fold CV)
- Trained on only 15 videos per fold
- Handles different jersey colors and game scenarios

## Technical Highlights
- Smart data augmentation
- Jersey color metadata integration
- Efficient architecture for limited compute (Colab free GPU)

## Installation
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Usage
See `notebooks/model_training.ipynb` for complete training pipeline.

## Future Work
- Expand to all NBA teams
- Expand dataset
- Real-time inference optimization
