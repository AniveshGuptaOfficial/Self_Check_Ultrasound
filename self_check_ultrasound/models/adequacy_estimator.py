"""
Task-Specific Adequacy Estimator (Module M4).
Estimates whether an image's quality and acoustic representations are sufficient
for reliable downstream cardiac view analysis.

Two modes:
- FITTED: logistic regression trained by scripts/train_adequacy.py on held-out
  (out-of-fold, sequence-group-aware) downstream outcomes, then saved with .save().
- HEURISTIC (is_fitted == False): fixed fallback 0.4*quality/9 + 0.6*confidence. This is NOT
  a learned or calibrated probability; results produced in this mode must be labelled as such.
"""

from pathlib import Path
from typing import Union
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy


class AdequacyEstimator:
    """
    Predicts binary task-specific adequacy P(Adequate | Features).
    Adequacy target (see scripts/train_adequacy.py): the downstream view classifier
    predicts the correct standard cardiac view for the frame (and true view is not Random).
    """

    FEATURE_NAMES = ("predicted_quality", "top1_confidence", "top2_margin", "entropy")

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

        safe_probs = np.clip(view_probabilities, 1e-7, 1.0)
        ent = entropy(safe_probs, axis=1).reshape(-1, 1)

        q_feat = predicted_quality.reshape(-1, 1)

        return np.hstack([q_feat, conf, margin, ent])

    def fit(self, X_features: np.ndarray, y_adequacy_targets: np.ndarray) -> None:
        """Fits the adequacy model on held-out (out-of-fold) adequacy targets."""
        X_scaled = self.scaler.fit_transform(X_features)
        self.clf.fit(X_scaled, y_adequacy_targets)
        self.is_fitted = True

    def predict_probability(self, X_features: np.ndarray) -> np.ndarray:
        """
        Probability of adequacy P(Adequate = 1) (fitted mode), or the heuristic score
        (unfitted mode; not calibrated).
        """
        if not self.is_fitted:
            q_norm = np.clip(X_features[:, 0] / 9.0, 0.0, 1.0)
            conf = X_features[:, 1]
            return 0.4 * q_norm + 0.6 * conf

        X_scaled = self.scaler.transform(X_features)
        return self.clf.predict_proba(X_scaled)[:, 1]

    def is_adequate(self, X_features: np.ndarray) -> np.ndarray:
        probs = self.predict_probability(X_features)
        return probs >= self.adequacy_threshold

    # ---- persistence (sklearn objects only; load with joblib from trusted local files) ----
    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"scaler": self.scaler, "clf": self.clf, "threshold": self.adequacy_threshold,
             "is_fitted": self.is_fitted},
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AdequacyEstimator":
        state = joblib.load(path)
        est = cls(adequacy_threshold=state["threshold"])
        est.scaler = state["scaler"]
        est.clf = state["clf"]
        est.is_fitted = state["is_fitted"]
        return est
