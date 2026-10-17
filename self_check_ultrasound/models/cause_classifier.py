"""
Cause Classifier (Module M5).
Identifies acoustic and operator acquisition faults:
- under_gain (acoustic attenuation)
- over_gain (acoustic saturation / blooming)
- acoustic_shadow (rib/calcification drop-out)
- motion_blur (probe jitter / respiratory motion)
- off_axis (anatomical misalignment)
- none (nominal)

Extracts physics-based indicators masked strictly to the active ultrasound sector
(excluding black background telemetry and fan margins).
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class CauseClassifier:
    """
    Classifier for acoustic artifact and acquisition parameter faults.
    Extracts sector-masked ultrasound physics metrics.
    """

    FAULT_CLASSES = (
        "none",
        "under_gain",
        "over_gain",
        "acoustic_shadow",
        "motion_blur",
        "off_axis",
    )

    def __init__(self, min_confidence: float = 0.60):
        self.min_confidence = min_confidence
        self.scaler = StandardScaler()
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        self.is_fitted = False

    def extract_physics_features(self, image: Image.Image) -> np.ndarray:
        """
        Extracts sector-masked ultrasound physics indicators:
        - Sector luminosity & dynamic range
        - Saturation ratio (blooming)
        - Inner sector high-frequency gradient energy (sharpness/blur)
        - Localized central column drop-out and minimum (shadowing)
        - Fan apex offset and lateral asymmetry (off-axis misalignment)
        - Sector coverage and contrast spread
        """
        gray = image.convert("L").resize((128, 128))
        arr = np.array(gray, dtype=np.float32) / 255.0

        # Mask out black ultrasound telemetry & fan margins (background < 0.05)
        sector_mask = arr > 0.05
        if np.sum(sector_mask) < 100:
            # Completely uninterpretable / black image
            return np.zeros(10, dtype=np.float32)

        sector_pixels = arr[sector_mask]
        mean_lum = float(np.mean(sector_pixels))
        p10 = float(np.percentile(sector_pixels, 10))
        p90 = float(np.percentile(sector_pixels, 90))

        # Saturation ratio: pixels blown out to white (> 0.90) inside sector
        sat_ratio = float(np.mean(sector_pixels > 0.90))

        # Inner sharpness indicator: eroded by 3 pixels to avoid sector border jump artifacts
        inner = np.zeros_like(sector_mask)
        inner[3:-3, 3:-3] = (
            sector_mask[3:-3, 3:-3] & sector_mask[:-6, 3:-3] & sector_mask[6:, 3:-3] &
            sector_mask[3:-3, :-6] & sector_mask[3:-3, 6:]
        )
        diff_x = np.abs(arr[:, 1:] - arr[:, :-1])
        diff_y = np.abs(arr[1:, :] - arr[:-1, :])
        grad = diff_x[:-1, :] + diff_y[:, :-1]
        inner_grad = grad[inner[:-1, :-1]]
        sharpness = float(np.mean(inner_grad)) if len(inner_grad) > 0 else 0.0

        # Mid-depth column drop-out and central profile dip (for acoustic shadows)
        col_counts = np.sum(sector_mask, axis=0)
        act_cols = np.where(col_counts > 30)[0]
        if len(act_cols) > 20:
            c1 = act_cols[int(len(act_cols) * 0.2)]
            c2 = act_cols[int(len(act_cols) * 0.8)]
            col_prof = np.mean(arr[32:96, c1:c2], axis=0)
            col_drop = float(np.max(col_prof) - np.min(col_prof))
            col_min = float(np.min(col_prof))
        else:
            col_min, col_drop = 0.0, 0.0

        # Sector coverage ratio
        coverage = float(np.mean(sector_mask))

        # Apex offset and lateral asymmetry (for off-axis / misalignment)
        top_rows = sector_mask[5:25, :]
        top_cols = np.where(np.sum(top_rows, axis=0) > 2)[0]
        apex_center = float(np.mean(top_cols)) if len(top_cols) > 0 else 64.0
        apex_offset = float(abs(apex_center - 64.0))

        left_cov = np.sum(sector_mask[:, :64])
        right_cov = np.sum(sector_mask[:, 64:])
        asymmetry = float(abs(left_cov - right_cov) / (left_cov + right_cov + 1e-6))

        return np.array([
            mean_lum,
            sat_ratio,
            p90,
            sharpness,
            col_drop,
            col_min,
            coverage,
            apex_offset,
            asymmetry,
            float(p90 - p10),  # contrast spread
        ], dtype=np.float32)

    def fit(self, X_features: np.ndarray, y_labels: np.ndarray) -> None:
        """
        Fits the random forest classifier on standardized physics features.
        """
        X_scaled = self.scaler.fit_transform(X_features)
        self.clf.fit(X_scaled, y_labels)
        self.is_fitted = True

    def train_on_reference_samples(
        self,
        splits_df: Any,
        dataset_root: Path,
        n_samples_per_view: int = 6,
        random_seed: int = 42,
    ) -> None:
        """
        Calibrates the classifier on reference training echocardiograms augmented with
        controlled acoustic degradations.
        """
        from ..data.degradations import SyntheticDegradationPipeline

        pipe = SyntheticDegradationPipeline(random_seed=random_seed)
        train_rows = (
            splits_df[
                (splits_df["split"] == "train")
                & (splits_df["quality_grade"] >= 7)
            ]
            .groupby("view_class", group_keys=False)
            .apply(lambda g: g.head(n_samples_per_view), include_groups=False if hasattr(splits_df.groupby("view_class"), "apply") else True)
        )

        X_list, y_list = [], []
        for _, r in train_rows.iterrows():
            img_path = dataset_root / r["relative_path"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")

            for fault in self.FAULT_CLASSES:
                sev = 0.0 if fault == "none" else (0.90 if fault == "off_axis" else 0.70)
                deg_img, _ = pipe.apply(img, fault, severity=sev)
                feats = self.extract_physics_features(deg_img)
                X_list.append(feats)
                y_list.append(fault)

        if len(X_list) > 0:
            self.fit(np.array(X_list), np.array(y_list))

    def predict(self, X_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          pred_classes: array of fault strings
          confidences: array of top-1 probabilities
          can_advise: boolean array indicating confidence >= min_confidence
        """
        if not self.is_fitted:
            preds = []
            confs = []
            can_adv = []
            for feat in X_features:
                lum, sat, p90, sharp, dip, min_c, cov, apex_offset, asym, spread = feat

                if cov < 0.15 or apex_offset > 12.0 or asym > 0.45:
                    pred, conf = "off_axis", 0.85
                elif sat > 0.15 or lum > 0.55:
                    pred, conf = "over_gain", 0.90
                elif lum < 0.25 or p90 < 0.38:
                    pred, conf = "under_gain", 0.88
                elif min_c <= 0.20 and dip >= 0.35 and sat < 0.02:
                    pred, conf = "acoustic_shadow", 0.80
                elif sharp < 0.138 and sat < 0.02:
                    pred, conf = "motion_blur", 0.85
                elif sharp >= 0.145 and min_c >= 0.22 and dip <= 0.33 and 0.33 <= lum <= 0.48:
                    pred, conf = "none", 0.85
                else:
                    pred, conf = "off_axis", 0.70

                preds.append(pred)
                confs.append(conf)
                can_adv.append(conf >= self.min_confidence)

            return np.array(preds), np.array(confs), np.array(can_adv)

        X_scaled = self.scaler.transform(X_features)
        probs = self.clf.predict_proba(X_scaled)
        top_idx = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        pred_classes = np.array([self.clf.classes_[i] for i in top_idx])
        can_advise = confs >= self.min_confidence

        return pred_classes, confs, can_advise
