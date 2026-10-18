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

Evidence levels (be explicit when reporting results):
- evaluate_on_synthetic(): accuracy on frames degraded by OUR synthetic pipeline, using
  held-out sequence groups. This measures separability of the synthetic faults only.
- evaluate_against_expert_grades(): AUROC of P(none) against REAL expert quality grades.
  This is the only check of the classifier against real (non-synthetic) quality variation.
"""

from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
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

    # ------------------------------------------------------------------ features
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
            return np.zeros(10, dtype=np.float32)

        sector_pixels = arr[sector_mask]
        mean_lum = float(np.mean(sector_pixels))
        p10 = float(np.percentile(sector_pixels, 10))
        p90 = float(np.percentile(sector_pixels, 90))

        sat_ratio = float(np.mean(sector_pixels > 0.90))

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

        coverage = float(np.mean(sector_mask))

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

    # ------------------------------------------------------------------ fitting
    def fit(self, X_features: np.ndarray, y_labels: np.ndarray) -> None:
        """
        Fits the random forest classifier on standardized physics features.
        """
        X_scaled = self.scaler.fit_transform(X_features)
        self.clf.fit(X_scaled, y_labels)
        self.is_fitted = True

    @staticmethod
    def _select_reference_rows(
        splits_df: pd.DataFrame,
        split: str,
        n_per_view: int,
        grade_min: Optional[float] = None,
        grade_max: Optional[float] = None,
        seed: int = 42,
    ) -> List[pd.Series]:
        """
        Picks up to n_per_view frames per view from `split`, spreading the picks round-robin
        across DIFFERENT sequence groups so the sample is not made of adjacent frames of one
        acquisition run.
        """
        rng = np.random.RandomState(seed)
        sub = splits_df[splits_df["split"] == split]
        if grade_min is not None:
            sub = sub[sub["quality_grade"] >= grade_min]
        if grade_max is not None:
            sub = sub[sub["quality_grade"] <= grade_max]

        rows: List[pd.Series] = []
        for view in sorted(sub["view_class"].unique()):
            vdf = sub[sub["view_class"] == view]
            groups = sorted(vdf["sequence_group"].unique())
            rng.shuffle(groups)
            per_group = {
                g: vdf[vdf["sequence_group"] == g].sample(frac=1.0, random_state=seed)
                for g in groups
            }
            cursor = {g: 0 for g in groups}
            picked: List[pd.Series] = []
            while len(picked) < n_per_view:
                progressed = False
                for g in groups:
                    if len(picked) >= n_per_view:
                        break
                    if cursor[g] < len(per_group[g]):
                        picked.append(per_group[g].iloc[cursor[g]])
                        cursor[g] += 1
                        progressed = True
                if not progressed:
                    break
            rows.extend(picked)
        return rows

    def train_on_reference_samples(
        self,
        splits_df: Any,
        dataset_root: Path,
        n_samples_per_view: int = 6,
        random_seed: int = 42,
        split: str = "train",
    ) -> int:
        """
        Calibrates the classifier on reference echocardiograms (grade >= 7) from `split`
        augmented with controlled synthetic degradations. Returns the number of training
        samples used.
        """
        from ..data.degradations import SyntheticDegradationPipeline

        pipe = SyntheticDegradationPipeline(random_seed=random_seed)
        rows = self._select_reference_rows(
            splits_df, split, n_samples_per_view, grade_min=7, seed=random_seed
        )

        X_list, y_list = [], []
        for r in rows:
            img_path = Path(dataset_root) / r["relative_path"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")

            for fault in self.FAULT_CLASSES:
                sev = 0.0 if fault == "none" else (0.90 if fault == "off_axis" else 0.70)
                deg_img, _ = pipe.apply(img, fault, severity=sev)
                X_list.append(self.extract_physics_features(deg_img))
                y_list.append(fault)

        if len(X_list) > 0:
            self.fit(np.array(X_list), np.array(y_list))
        return len(X_list)

    # ------------------------------------------------------------------ inference
    def _rule_predict(self, X_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        preds, confs, can_adv = [], [], []
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

    def predict_proba(self, X_features: np.ndarray) -> np.ndarray:
        """
        Class-probability matrix with columns ordered as FAULT_CLASSES.
        Fitted model: random-forest probabilities. Unfitted: rule-based top class gets its
        rule confidence and the remainder is spread evenly over the other classes.
        """
        n = len(X_features)
        out = np.zeros((n, len(self.FAULT_CLASSES)), dtype=np.float64)
        if not self.is_fitted:
            preds, confs, _ = self._rule_predict(X_features)
            for i, (p, c) in enumerate(zip(preds, confs)):
                out[i, :] = (1.0 - c) / (len(self.FAULT_CLASSES) - 1)
                out[i, self.FAULT_CLASSES.index(p)] = c
            return out

        probs = self.clf.predict_proba(self.scaler.transform(X_features))
        for j, cls in enumerate(self.clf.classes_):
            out[:, self.FAULT_CLASSES.index(str(cls))] = probs[:, j]
        return out

    def predict(self, X_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          pred_classes: array of fault strings
          confidences: array of top-1 probabilities
          can_advise: boolean array indicating confidence >= min_confidence
        """
        if not self.is_fitted:
            return self._rule_predict(X_features)

        probs = self.predict_proba(X_features)
        top_idx = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        pred_classes = np.array([self.FAULT_CLASSES[i] for i in top_idx])
        return pred_classes, confs, confs >= self.min_confidence

    # ------------------------------------------------------------------ evaluation
    def evaluate_on_synthetic(
        self,
        splits_df: pd.DataFrame,
        dataset_root: Path,
        split: str = "val",
        n_samples_per_view: int = 5,
        severities: Tuple[float, ...] = (0.3, 0.5, 0.7, 0.9),
        seed: int = 123,
    ) -> Dict[str, Any]:
        """
        Held-out evaluation on SYNTHETIC faults: base frames come from `split` (disjoint
        sequence groups from the training split), degradations use a different RNG seed.
        Reports overall / per-fault / per-severity accuracy, abstention rate and confusion.
        """
        from ..data.degradations import SyntheticDegradationPipeline

        pipe = SyntheticDegradationPipeline(random_seed=seed)
        rows = self._select_reference_rows(splits_df, split, n_samples_per_view, grade_min=7, seed=seed)

        y_true: List[str] = []
        y_pred: List[str] = []
        sevs: List[float] = []
        abstain: List[bool] = []

        for r in rows:
            img_path = Path(dataset_root) / r["relative_path"]
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert("RGB")
            for fault in self.FAULT_CLASSES:
                for sev in ((0.0,) if fault == "none" else severities):
                    deg_img, _ = pipe.apply(img, fault, severity=max(sev, 0.1))
                    feats = self.extract_physics_features(deg_img).reshape(1, -1)
                    pred, conf, can_adv = self.predict(feats)
                    y_true.append(fault)
                    y_pred.append(str(pred[0]))
                    sevs.append(sev)
                    abstain.append(not bool(can_adv[0]))

        if not y_true:
            return {"n": 0, "note": "no images found"}

        yt, yp, sv = np.array(y_true), np.array(y_pred), np.array(sevs)
        correct = yt == yp
        per_fault = {f: float(np.mean(correct[yt == f])) for f in self.FAULT_CLASSES if np.any(yt == f)}
        per_sev = {f"{s:.1f}": float(np.mean(correct[sv == s])) for s in sorted(set(sevs)) if s > 0}
        confusion = {
            t: {p: int(np.sum((yt == t) & (yp == p))) for p in self.FAULT_CLASSES}
            for t in self.FAULT_CLASSES if np.any(yt == t)
        }
        return {
            "evidence": "synthetic faults, held-out sequence groups",
            "split": split,
            "n": int(len(yt)),
            "accuracy": float(np.mean(correct)),
            "per_fault_accuracy": per_fault,
            "per_severity_accuracy": per_sev,
            "abstain_rate": float(np.mean(abstain)),
            "confusion": confusion,
        }

    def evaluate_against_expert_grades(
        self,
        splits_df: pd.DataFrame,
        dataset_root: Path,
        split: str = "val",
        n_per_view: int = 20,
        high_min: float = 7,
        low_max: float = 3,
        seed: int = 123,
    ) -> Dict[str, Any]:
        """
        REAL-data check: how well does P(none) separate expert high-grade frames (>= high_min)
        from expert low-grade frames (<= low_max)? Reports AUROC and the fraction of each
        group predicted as 'none'. No synthetic degradation is involved.
        """
        high_rows = self._select_reference_rows(splits_df, split, n_per_view, grade_min=high_min, seed=seed)
        low_rows = self._select_reference_rows(splits_df, split, n_per_view, grade_max=low_max, seed=seed)

        scores, labels, preds = [], [], []
        for rows, label in ((high_rows, 1), (low_rows, 0)):
            for r in rows:
                img_path = Path(dataset_root) / r["relative_path"]
                if not img_path.exists():
                    continue
                img = Image.open(img_path).convert("RGB")
                feats = self.extract_physics_features(img).reshape(1, -1)
                p = self.predict_proba(feats)[0]
                scores.append(float(p[self.FAULT_CLASSES.index("none")]))
                preds.append(self.FAULT_CLASSES[int(np.argmax(p))])
                labels.append(label)

        labels_a, scores_a, preds_a = np.array(labels), np.array(scores), np.array(preds)
        result: Dict[str, Any] = {
            "evidence": "real expert grades (no synthetic degradation)",
            "split": split,
            "n_high": int(np.sum(labels_a == 1)),
            "n_low": int(np.sum(labels_a == 0)),
        }
        if len(set(labels)) == 2:
            result["auroc_p_none_vs_high_grade"] = float(roc_auc_score(labels_a, scores_a))
            result["frac_predicted_none_high_grade"] = float(np.mean(preds_a[labels_a == 1] == "none"))
            result["frac_predicted_none_low_grade"] = float(np.mean(preds_a[labels_a == 0] == "none"))
        else:
            result["note"] = "need both high- and low-grade frames in this split"
        return result
