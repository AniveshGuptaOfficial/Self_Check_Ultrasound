"""
Task-Specific Adequacy Estimator (Module M4).
Estimates whether an image's quality and acoustic representations are sufficient
for reliable downstream cardiac view analysis.

Trained on held-out out-of-fold downstream performance to prevent in-sample overfitting.
"""

from typing import Dict, Any, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy


class AdequacyEstimator:
    """
    Predicts binary task-specific adequacy P(Adequate | Features).
    Adequacy is defined by whether the downstream model predicts the correct cardiac view
    with high confidence and low entropy.
    """

    def __init__(self, adequacy_threshold: float = 0.70):
        self.adequacy_threshold = adequacy_threshold
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(C=1.0, random_state=42, max_iter=500)
        self.is_fitted = False

    def extract_adequacy_features(
        self,
        predicted_quality: np.ndarray,
        view_probabilities: np.ndarray,
    ) -> np.ndarray:
        """
        Extracts reliability features combining:
        1. Predicted quality grade (0-9)
        2. Top-1 view confidence (max p)
        3. Top-2 margin (p_top1 - p_top2)
        4. Predictive Shannon entropy
        """
        if view_probabilities.ndim == 1:
            view_probabilities = view_probabilities.reshape(1, -1)
        if np.isscalar(predicted_quality):
            predicted_quality = np.array([predicted_quality])

        conf = np.max(view_probabilities, axis=1, keepdims=True)
        sorted_probs = np.sort(view_probabilities, axis=1)
        margin = (sorted_probs[:, -1] - sorted_probs[:, -2]).reshape(-1, 1)

        # Numerical safety for entropy
        safe_probs = np.clip(view_probabilities, 1e-7, 1.0)
        ent = entropy(safe_probs, axis=1).reshape(-1, 1)

        q_feat = predicted_quality.reshape(-1, 1)

        return np.hstack([q_feat, conf, margin, ent])

    def fit(self, X_features: np.ndarray, y_adequacy_targets: np.ndarray) -> None:
        """
        Fits the adequacy model on out-of-fold validation targets.
        """
        X_scaled = self.scaler.fit_transform(X_features)
        self.clf.fit(X_scaled, y_adequacy_targets)
        self.is_fitted = True

    def predict_probability(self, X_features: np.ndarray) -> np.ndarray:
        """
        Outputs calibrated probability of adequacy P(Adequate = 1).
        """
        if not self.is_fitted:
            # Heuristic fallback before fitting: calibrated combination of quality and confidence
            # Quality scaled to [0, 1] + Confidence
            q_norm = np.clip(X_features[:, 0] / 9.0, 0.0, 1.0)
            conf = X_features[:, 1]
            return 0.4 * q_norm + 0.6 * conf

        X_scaled = self.scaler.transform(X_features)
        return self.clf.predict_proba(X_scaled)[:, 1]

    def is_adequate(self, X_features: np.ndarray) -> np.ndarray:
        probs = self.predict_probability(X_features)
        return probs >= self.adequacy_threshold
