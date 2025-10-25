"""
Model architecture for OOB Detection
MobileNetV2 + LSTM with jersey color embedding
"""

import torch
import torch.nn as nn
from torchvision import models

from config import (
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    COLOR_EMBEDDING_SIZE
)


class OOBModelWithColor(nn.Module):
    """Model that uses both video frames and jersey color metadata"""
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        # MobileNet feature extractor
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(mobilenet.children())[:-1])
        
        # Freeze early layers (keep last 10 trainable)
        for param in list(self.feature_extractor.parameters())[:-10]:
            param.requires_grad = False
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=1280,  # MobileNetV2 output size
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=LSTM_DROPOUT
        )
        
        # Jersey color embedding
        self.color_embedding = nn.Sequential(
            nn.Linear(3, COLOR_EMBEDDING_SIZE),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Combined classifier
        self.classifier = nn.Sequential(
            nn.Linear(LSTM_HIDDEN_SIZE + COLOR_EMBEDDING_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, frames, color):
        """
        Forward pass
        
        Args:
            frames: (batch, time, channels, height, width)
            color: (batch, 3) - one-hot encoded jersey color
            
        Returns:
            output: (batch, num_classes) - class logits
        """
        batch_size, time_steps = frames.size(0), frames.size(1)
        
        # Process frames through CNN
        x = frames.view(-1, frames.size(2), frames.size(3), frames.size(4))
        features = self.feature_extractor(x)
        features = features.mean(dim=[2, 3])  # Global average pooling
        features = features.view(batch_size, time_steps, -1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        final_features = lstm_out[:, -1, :]  # Use last timestep
        
        # Jersey color embedding
        color_features = self.color_embedding(color)
        
        # Concatenate and classify
        combined = torch.cat([final_features, color_features], dim=1)
        output = self.classifier(combined)
        return output
