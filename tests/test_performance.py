"""
Automated Unit Tests for Computational Performance Benchmarks (Phase 7).
Verifies that artifacts/performance_benchmarks.json contains valid schemas,
positive latency and throughput values, and meets clinical real-time bounds.
"""

import json
from pathlib import Path
import pytest
from self_check_ultrasound.config import default_config


def test_performance_benchmarks_artifact_exists_and_valid():
    benchmarks_path = default_config.paths.artifacts_dir / "performance_benchmarks.json"
    if not benchmarks_path.exists():
        pytest.skip("performance_benchmarks.json not generated yet. Run scripts/benchmark_performance.py first.")

    with open(benchmarks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Hardware section
    assert "hardware" in data
    hw = data["hardware"]
    assert "platform" in hw
    assert hw["logical_cores"] > 0
    assert hw["pytorch_threads"] > 0

    # 2. Models section
    assert "models" in data
    models = data["models"]
    assert "cardiac_echo_net" in models
    assert "quality_assessor_net" in models
    assert "cause_classifier" in models
    assert "adequacy_estimator" in models
    assert "reacquisition_policy" in models

    # Check CardiacEchoNet batch 1
    cen_b1 = models["cardiac_echo_net"]["batch_1"]
    assert cen_b1["mean_ms"] > 0
    assert cen_b1["fps"] > 0
    assert cen_b1["mean_ms"] < 250.0  # reasonable CPU bound

    # Check QualityAssessorNet batch 1
    qan_b1 = models["quality_assessor_net"]["batch_1"]
    assert qan_b1["mean_ms"] > 0
    assert qan_b1["fps"] > 0
    assert qan_b1["mean_ms"] < 250.0

    # Check CauseClassifier
    cc = models["cause_classifier"]
    assert cc["feature_extraction_ms"]["mean_ms"] > 0
    assert cc["rf_inference_ms"]["mean_ms"] > 0

    # Check AdequacyEstimator
    adeq = models["adequacy_estimator"]
    assert adeq["mean_ms"] > 0

    # Check Policy (microseconds)
    pol = models["reacquisition_policy"]
    assert pol["mean_us"] > 0

    # 3. End-to-End Decision Cycle
    assert "end_to_end" in data
    e2e = data["end_to_end"]
    assert e2e["mean_ms"] > 0
    assert e2e["fps"] > 0
    assert e2e["mean_ms"] < 500.0  # Must be interactive (< 500 ms per frame on CPU)

    # 4. Memory footprint
    assert "memory_profile" in data
    mem = data["memory_profile"]
    assert mem["total_peak_traced_mb"] > 0
    assert mem["total_peak_traced_mb"] < 100.0  # Must be lightweight
