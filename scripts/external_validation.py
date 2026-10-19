"""
External validation of the trained pipeline on a second echocardiography dataset.

Needs a manifest CSV that YOU build from the downloaded dataset (formats differ per dataset):
    relative_path   path of a PNG/JPG frame, relative to --root
    view_label      the dataset's own view label (e.g. PLAX, PSAX, A4C, A2C, Other)
    quality_grade   OPTIONAL numeric expert quality, higher = better (e.g. CAMUS poor=0, medium=1, good=2)
    patient_id      OPTIONAL, used only to report the number of patients

Supported label maps (see utils/external.py): tmed2, cameo, camus.
Requires artifacts/cardiac_echo_net.pt and quality_assessor_net.pt; uses artifacts/adequacy_estimator.joblib
if present (otherwise the heuristic score, which is labelled as such in the output).

Usage:
  python scripts/external_validation.py --dataset tmed2 --root /data/tmed2 --manifest /data/tmed2/manifest.csv
Writes artifacts/external_<dataset>.json
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from self_check_ultrasound.config import default_config
from self_check_ultrasound.utils.seed import set_seed
from self_check_ultrasound.utils.paths import resolve_safe_path
from self_check_ultrasound.utils.external import map_views, summarize, auroc_ci
from self_check_ultrasound.data.transforms import get_transforms
from self_check_ultrasound.models.view_classifier import CardiacEchoNet
from self_check_ultrasound.models.quality_assessor import QualityAssessorNet
from self_check_ultrasound.models.adequacy_estimator import AdequacyEstimator


class ManifestDataset(Dataset):
    def __init__(self, df, root, transform):
        self.df, self.root, self.transform = df.reset_index(drop=True), Path(root), transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        path = resolve_safe_path(self.root, self.df.loc[i, "relative_path"])
        img = Image.open(path).convert("RGB")  # greyscale frames are replicated to 3 channels
        return self.transform(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tmed2", "cameo", "camus"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all rows")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    set_seed(default_config.data.random_seed)
    art = default_config.paths.artifacts_dir
    device = torch.device(default_config.training.device)
    for need in ("cardiac_echo_net.pt", "quality_assessor_net.pt"):
        if not (art / need).exists():
            raise FileNotFoundError(f"{art / need} missing. Train the baselines first.")

    df = pd.read_csv(args.manifest)
    for col in ("relative_path", "view_label"):
        if col not in df.columns:
            raise ValueError(f"manifest needs a '{col}' column")
    if args.max_samples and len(df) > args.max_samples:
        df = df.sample(n=args.max_samples, random_state=42)
    df = df.reset_index(drop=True)
    groups = map_views(df["view_label"], args.dataset)
    print(f"{len(df)} frames; {int(groups.notna().sum())} map to a CACTUS class, "
          f"{int(groups.isna().sum())} are unseen views")

    view_model = CardiacEchoNet(num_classes=default_config.data.num_classes).to(device)
    view_model.load_state_dict(torch.load(art / "cardiac_echo_net.pt", map_location=device, weights_only=True))
    quality_model = QualityAssessorNet().to(device)
    quality_model.load_state_dict(torch.load(art / "quality_assessor_net.pt", map_location=device, weights_only=True))
    view_model.eval()
    quality_model.eval()

    est_path = art / "adequacy_estimator.joblib"
    est = AdequacyEstimator.load(est_path) if est_path.exists() else AdequacyEstimator()
    est.adequacy_threshold = default_config.policy.adequacy_threshold

    tf = get_transforms(is_train=False, image_size=default_config.data.image_size)
    loader = DataLoader(ManifestDataset(df, args.root, tf), batch_size=args.batch_size, shuffle=False, num_workers=0)
    probs_all, qual_all = [], []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            probs_all.append(torch.softmax(view_model(x), dim=1).cpu().numpy())
            qual_all.append(quality_model(x).cpu().numpy())
    probs = np.concatenate(probs_all)
    quality = np.clip(np.concatenate(qual_all), 0.0, 9.0)

    adequacy = est.predict_probability(est.extract_adequacy_features(quality, probs))
    results = {
        "dataset": args.dataset,
        "adequacy_mode": "fitted" if est.is_fitted else "HEURISTIC (not calibrated)",
        "view_and_adequacy": summarize(probs, groups, adequacy, est.adequacy_threshold),
        "note": "Domain shift: CACTUS is a phantom on one scanner. Report these numbers as external, not tuned.",
    }

    if "quality_grade" in df.columns:
        g = df["quality_grade"].astype(float).values
        rho = spearmanr(quality, g)
        top = g >= np.nanmax(g)
        results["quality_vs_expert"] = {
            "spearman_rho": float(rho.correlation), "p_value": float(rho.pvalue),
            "auroc_top_grade_vs_rest": auroc_ci(top.astype(int), quality),
            "auroc_adequacy_top_grade_vs_rest": auroc_ci(top.astype(int), adequacy),
        }

    out = art / f"external_{args.dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
