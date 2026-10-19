"""
Computational Performance Benchmarking Script (Phase 7).
Profiles inference latency, throughput (FPS), dataset loading speed,
and peak memory consumption across all SelfCheckUltrasound pipeline models.

Writes: artifacts/performance_benchmarks.json

Usage:
  python scripts/benchmark_performance.py [--iterations 50] [--warmup 10]
"""

import sys
import os
import time
import json
import argparse
import platform
import tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from self_check_ultrasound.config import default_config
from self_check_ultrasound.utils.seed import set_seed
from self_check_ultrasound.data.transforms import get_transforms
from self_check_ultrasound.models.view_classifier import CardiacEchoNet
from self_check_ultrasound.models.quality_assessor import QualityAssessorNet
from self_check_ultrasound.models.adequacy_estimator import AdequacyEstimator
from self_check_ultrasound.models.cause_classifier import CauseClassifier
from self_check_ultrasound.guidance.reacquisition_policy import ReacquisitionPolicy


def compute_timing_stats(latencies_ms):
    arr = np.array(latencies_ms)
    mean_ms = float(np.mean(arr))
    median_ms = float(np.median(arr))
    std_ms = float(np.std(arr))
    p95_ms = float(np.percentile(arr, 95))
    p99_ms = float(np.percentile(arr, 99))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return {
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(median_ms, 3),
        "std_ms": round(std_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "p99_ms": round(p99_ms, 3),
        "fps": round(fps, 2),
    }


def benchmark_nn_model(model, input_tensor, warmup=10, iterations=50, device="cpu"):
    model.eval()
    input_tensor = input_tensor.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
            
    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            
    return compute_timing_stats(latencies)


def main():
    parser = argparse.ArgumentParser(description="SelfCheckUltrasound: Computational Performance Benchmarking")
    parser.add_argument("--iterations", type=int, default=50, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 8, 32], help="Batch sizes to benchmark")
    args = parser.parse_args()

    set_seed(42)
    device = "cpu"
    artifacts_dir = default_config.paths.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SelfCheckUltrasound: Computational Performance Benchmarking (Phase 7)")
    print("=" * 72)

    # 1. Hardware & Environment Profiling
    hw_info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cores": os.cpu_count(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "pytorch_threads": torch.get_num_threads(),
        "device": device,
    }
    print(f"OS Platform:       {hw_info['platform']}")
    print(f"Processor:         {hw_info['processor']}")
    print(f"Logical Cores:     {hw_info['logical_cores']}")
    print(f"PyTorch Version:   {hw_info['pytorch_version']} (threads: {hw_info['pytorch_threads']})")
    print("-" * 72)

    tracemalloc.start()
    baseline_mem_kb, _ = tracemalloc.get_traced_memory()

    # 2. Load Checkpoints & Models
    print("Loading models and checkpoints...")
    view_model = CardiacEchoNet(num_classes=default_config.data.num_classes).to(device)
    view_ckpt = artifacts_dir / "cardiac_echo_net.pt"
    if view_ckpt.exists():
        view_model.load_state_dict(torch.load(view_ckpt, map_location=device, weights_only=True))
        print("  - Loaded CardiacEchoNet from artifacts/cardiac_echo_net.pt")
    else:
        print("  - Using initialized CardiacEchoNet (checkpoint not found)")

    quality_model = QualityAssessorNet().to(device)
    quality_ckpt = artifacts_dir / "quality_assessor_net.pt"
    if quality_ckpt.exists():
        quality_model.load_state_dict(torch.load(quality_ckpt, map_location=device, weights_only=True))
        print("  - Loaded QualityAssessorNet from artifacts/quality_assessor_net.pt")
    else:
        print("  - Using initialized QualityAssessorNet (checkpoint not found)")

    adequacy_estimator = AdequacyEstimator()
    adequacy_ckpt = artifacts_dir / "adequacy_estimator.joblib"
    if adequacy_ckpt.exists():
        adequacy_estimator = AdequacyEstimator.load(adequacy_ckpt)
        print("  - Loaded AdequacyEstimator from artifacts/adequacy_estimator.joblib")
    else:
        print("  - Using fallback heuristic AdequacyEstimator")

    cause_classifier = CauseClassifier(min_confidence=0.60)
    policy = ReacquisitionPolicy()
    transform = get_transforms(is_train=False)

    post_load_mem_kb, peak_load_mem_kb = tracemalloc.get_traced_memory()
    print(f"Memory footprint after model loading: {round(post_load_mem_kb / (1024 * 1024), 2)} MB (Peak: {round(peak_load_mem_kb / (1024 * 1024), 2)} MB)")
    print("-" * 72)

    benchmarks = {
        "hardware": hw_info,
        "models": {},
        "end_to_end": {},
        "data_throughput": {},
        "memory_profile": {},
    }

    # 3. Benchmark CardiacEchoNet (M1 / M2) across batch sizes
    print("Benchmarking CardiacEchoNet (Modules M1 / M2: View Classification)...")
    benchmarks["models"]["cardiac_echo_net"] = {}
    for bs in args.batch_sizes:
        dummy_input = torch.randn(bs, 3, 224, 224)
        stats = benchmark_nn_model(view_model, dummy_input, warmup=args.warmup, iterations=args.iterations, device=device)
        per_frame_ms = round(stats["mean_ms"] / bs, 3)
        throughput_fps = round(1000.0 / per_frame_ms, 2)
        benchmarks["models"]["cardiac_echo_net"][f"batch_{bs}"] = {
            **stats,
            "per_frame_ms": per_frame_ms,
            "throughput_fps": throughput_fps,
        }
        print(f"  Batch {bs:2d}: Mean = {stats['mean_ms']:6.2f} ms | Per-frame = {per_frame_ms:6.2f} ms | Throughput = {throughput_fps:6.1f} FPS")

    # 4. Benchmark QualityAssessorNet (M3) across batch sizes
    print("\nBenchmarking QualityAssessorNet (Module M3: Quality Assessment)...")
    benchmarks["models"]["quality_assessor_net"] = {}
    for bs in args.batch_sizes:
        dummy_input = torch.randn(bs, 3, 224, 224)
        stats = benchmark_nn_model(quality_model, dummy_input, warmup=args.warmup, iterations=args.iterations, device=device)
        per_frame_ms = round(stats["mean_ms"] / bs, 3)
        throughput_fps = round(1000.0 / per_frame_ms, 2)
        benchmarks["models"]["quality_assessor_net"][f"batch_{bs}"] = {
            **stats,
            "per_frame_ms": per_frame_ms,
            "throughput_fps": throughput_fps,
        }
        print(f"  Batch {bs:2d}: Mean = {stats['mean_ms']:6.2f} ms | Per-frame = {per_frame_ms:6.2f} ms | Throughput = {throughput_fps:6.1f} FPS")

    # 5. Benchmark CauseClassifier (M5)
    print("\nBenchmarking CauseClassifier (Module M5: Physics Features & Random Forest)...")
    dummy_full_hd = Image.new("RGB", (1920, 1080), color=(30, 30, 30))
    # Warmup feature extraction
    for _ in range(3):
        _ = cause_classifier.extract_physics_features(dummy_full_hd)
    
    feat_latencies = []
    for _ in range(min(args.iterations, 20)):
        t0 = time.perf_counter()
        feats = cause_classifier.extract_physics_features(dummy_full_hd)
        t1 = time.perf_counter()
        feat_latencies.append((t1 - t0) * 1000.0)
    feat_stats = compute_timing_stats(feat_latencies)
    print(f"  Physics Feature Extraction (1920x1080): Mean = {feat_stats['mean_ms']:6.2f} ms | Median = {feat_stats['median_ms']:6.2f} ms")

    dummy_feats_row = feats.reshape(1, -1)
    rf_latencies = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        _ = cause_classifier.predict(dummy_feats_row)
        t1 = time.perf_counter()
        rf_latencies.append((t1 - t0) * 1000.0)
    rf_stats = compute_timing_stats(rf_latencies)
    print(f"  Random Forest Fault Inference:        Mean = {rf_stats['mean_ms']:6.3f} ms | Median = {rf_stats['median_ms']:6.3f} ms")

    benchmarks["models"]["cause_classifier"] = {
        "feature_extraction_ms": feat_stats,
        "rf_inference_ms": rf_stats,
        "total_module_ms": round(feat_stats["mean_ms"] + rf_stats["mean_ms"], 3),
    }

    # 6. Benchmark AdequacyEstimator (M4)
    print("\nBenchmarking AdequacyEstimator (Module M4: Task Adequacy Prediction)...")
    dummy_probs = np.array([[0.1, 0.7, 0.05, 0.05, 0.05, 0.05]])
    adeq_latencies = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        adeq_feats = adequacy_estimator.extract_adequacy_features(np.array([7.2]), dummy_probs)
        _ = adequacy_estimator.predict_probability(adeq_feats)
        t1 = time.perf_counter()
        adeq_latencies.append((t1 - t0) * 1000.0)
    adeq_stats = compute_timing_stats(adeq_latencies)
    print(f"  Feature Extraction & Regressor:       Mean = {adeq_stats['mean_ms']:6.3f} ms | Median = {adeq_stats['median_ms']:6.3f} ms")
    benchmarks["models"]["adequacy_estimator"] = adeq_stats

    # 7. Benchmark ReacquisitionPolicy (M6 / M7)
    print("\nBenchmarking ReacquisitionPolicy (Modules M6 / M7: Triage & Guidance)...")
    policy_latencies_us = []
    for _ in range(args.iterations * 10):
        t0 = time.perf_counter()
        _ = policy.evaluate(adequacy_prob=0.45, diagnosed_fault="under_gain", fault_confidence=0.88, attempt_count=1)
        t1 = time.perf_counter()
        policy_latencies_us.append((t1 - t0) * 1e6)
    policy_stats_us = {
        "mean_us": round(float(np.mean(policy_latencies_us)), 2),
        "median_us": round(float(np.median(policy_latencies_us)), 2),
        "std_us": round(float(np.std(policy_latencies_us)), 2),
    }
    print(f"  Guarded Triage Decision:              Mean = {policy_stats_us['mean_us']:6.2f} us | Median = {policy_stats_us['median_us']:6.2f} us")
    benchmarks["models"]["reacquisition_policy"] = policy_stats_us

    # 8. End-to-End Decision Pipeline Latency
    print("\nBenchmarking End-to-End Decision Cycle (Full Single-Frame Pipeline)...")
    print("  [Raw Image -> Transform -> View M1 -> Quality M3 -> Adequacy M4 -> Cause M5 -> Policy M7]")
    
    # Warmup end-to-end
    for _ in range(3):
        x = transform(dummy_full_hd).unsqueeze(0).to(device)
        with torch.no_grad():
            v_probs = torch.softmax(view_model(x), dim=1).cpu().numpy()
            q_grade = float(quality_model(x).cpu().numpy().reshape(-1)[0])
        f_vec = cause_classifier.extract_physics_features(dummy_full_hd).reshape(1, -1)
        fault, conf, _ = cause_classifier.predict(f_vec)
        adeq_feats = adequacy_estimator.extract_adequacy_features(np.array([q_grade]), v_probs)
        p_adeq = float(adequacy_estimator.predict_probability(adeq_feats)[0])
        _ = policy.evaluate(p_adeq, fault[0], conf[0], attempt_count=1)

    e2e_latencies = []
    for _ in range(min(args.iterations, 25)):
        t0 = time.perf_counter()
        # Step A: Transform
        x = transform(dummy_full_hd).unsqueeze(0).to(device)
        # Step B: Neural Nets (M1 View, M3 Quality)
        with torch.no_grad():
            v_probs = torch.softmax(view_model(x), dim=1).cpu().numpy()
            q_grade = float(quality_model(x).cpu().numpy().reshape(-1)[0])
        # Step C: Physics Extraction & Cause Classifier (M5)
        f_vec = cause_classifier.extract_physics_features(dummy_full_hd).reshape(1, -1)
        fault, conf, _ = cause_classifier.predict(f_vec)
        # Step D: Adequacy Estimation (M4)
        adeq_feats = adequacy_estimator.extract_adequacy_features(np.array([q_grade]), v_probs)
        p_adeq = float(adequacy_estimator.predict_probability(adeq_feats)[0])
        # Step E: Guarded Triage Policy (M6/M7)
        _ = policy.evaluate(p_adeq, fault[0], conf[0], attempt_count=1)
        t1 = time.perf_counter()
        e2e_latencies.append((t1 - t0) * 1000.0)

    e2e_stats = compute_timing_stats(e2e_latencies)
    print(f"  Total End-to-End Latency:             Mean = {e2e_stats['mean_ms']:6.2f} ms | Median = {e2e_stats['median_ms']:6.2f} ms | P95 = {e2e_stats['p95_ms']:6.2f} ms")
    print(f"  End-to-End Frame Rate:                {e2e_stats['fps']:6.1f} FPS")
    benchmarks["end_to_end"] = e2e_stats

    # 9. Real Dataset Loading & Preprocessing Throughput
    manifest_path = artifacts_dir / "dataset_manifest.csv"
    if manifest_path.exists():
        print("\nBenchmarking Dataset Ingestion & Transform Throughput...")
        df_manifest = pd.read_csv(manifest_path, nrows=50)
        dataset_root = default_config.paths.dataset_dir
        
        load_latencies = []
        for _, row in df_manifest.iterrows():
            img_path = dataset_root / row["relative_path"]
            if img_path.exists():
                t0 = time.perf_counter()
                with Image.open(img_path) as img:
                    _ = transform(img.convert("RGB"))
                t1 = time.perf_counter()
                load_latencies.append((t1 - t0) * 1000.0)
                
        if load_latencies:
            load_stats = compute_timing_stats(load_latencies)
            print(f"  Disk Read + RGB Transform (Full HD):  Mean = {load_stats['mean_ms']:6.2f} ms | Throughput = {load_stats['fps']:6.1f} images/sec")
            benchmarks["data_throughput"] = load_stats

    # 10. Memory Consumption Profile
    _, final_peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mem_mb = round(final_peak_kb / (1024 * 1024), 2)
    benchmarks["memory_profile"] = {
        "model_weights_mb": round((post_load_mem_kb - baseline_mem_kb) / (1024 * 1024), 2),
        "total_peak_traced_mb": peak_mem_mb,
    }
    print(f"\nPeak Traced Heap Memory Footprint:      {peak_mem_mb} MB")

    # Save to artifacts/performance_benchmarks.json
    out_path = artifacts_dir / "performance_benchmarks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
    print("\n" + "=" * 72)
    print(f"Benchmark results successfully exported to:\n  {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
