"""
Image Quality Assessment Model (Module M3).
Predicts the expert quality grade (0-9) reflecting completeness and clarity.
"""

import torch
import torch.nn as nn
from .view_classifier import CardiacEchoNet


class QualityAssessorNet(nn.Module):
    """
    Quality estimation neural network.
    Uses the convolutional feature backbone to predict continuous quality grades (0 to 9).
    """

    def __init__(self, dropout_rate: float = 0.2):
        super().__init__()
        self.backbone = CardiacEchoNet(num_classes=6, dropout_rate=dropout_rate)
        # Replace the 6-class head with a scalar regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.features(x)
        pred_grade = self.regressor(feat)
        return pred_grade.squeeze(-1)
