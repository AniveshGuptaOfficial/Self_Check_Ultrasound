"""
Cardiac View Recognition Baseline Models (Module M1).
Includes:
- SimpleFeatureClassifier: Classical ML baseline using spatial intensity features.
- CardiacEchoNet: Compact, high-efficiency CNN baseline designed for reliable CPU execution.
"""

from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from PIL import Image


class SimpleFeatureClassifier:
    """
    Classical baseline: Extracts coarse grid intensity and histogram features,
    standardizes them, then fits a regularized Logistic Regression classifier.
    """

    def __init__(self, n_bins: int = 16, grid_size: int = 4):
        self.n_bins = n_bins
        self.grid_size = grid_size
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

    def extract_features(self, image_pil: Image.Image) -> np.ndarray:
        """
        Extracts spatial grid intensity averages and global histogram.
        """
        img_gray = image_pil.convert("L").resize((64, 64))
        arr = np.array(img_gray, dtype=np.float32) / 255.0

        # Global histogram
        hist, _ = np.histogram(arr, bins=self.n_bins, range=(0.0, 1.0), density=True)

        # Spatial grid features (grid_size x grid_size)
        gh, gw = arr.shape[0] // self.grid_size, arr.shape[1] // self.grid_size
        grid_means = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell = arr[i * gh : (i + 1) * gh, j * gw : (j + 1) * gw]
                grid_means.append(np.mean(cell))

        return np.concatenate([hist, np.array(grid_means, dtype=np.float32)])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)


class CardiacEchoNet(nn.Module):
    """
    Lightweight, deterministic CNN tailored for cardiac view recognition and local CPU inference.
    Architecture: 4 Conv-BN-ReLU-Pool stages followed by AdaptiveAvgPool and a Linear classifier.
    """

    def __init__(self, num_classes: int = 6, dropout_rate: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Stage 1: (3, 224, 224) -> (32, 112, 112)
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2: (32, 112, 112) -> (64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3: (64, 56, 56) -> (128, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 4: (128, 28, 28) -> (256, 14, 14)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.classifier(feat)
        return logits

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return torch.flatten(feat, 1)
