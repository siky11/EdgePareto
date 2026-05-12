#!/usr/bin/env python3
# convenience script that runs all onnx benchmarks automatically
# instead of typing every benchmark command manually, this scans models/onnx/
# and runs benchmark_onnx.py for each model in both fp32 and int8 mode

# Raspberry Pi:  python -m src.scripts.run_benchmarks_onnx --device cpu
# Jetson Nano:   python -m src.scripts.run_benchmarks_onnx --device cuda

import argparse
import subprocess
import sys
from pathlib import Path


ONNX_DIR = Path("models/onnx")  # subfolder structure: baseline/ standalone/ hybrid/
MODES = ["fp32", "int8"]  # both precisions for every model


if __name__ == "__main__":
    # --device is required so the script knows which hardware its running on
    # need to pass the right flag to benchmark_onnx.py
    parser = argparse.ArgumentParser(description="run all ONNX benchmarks")
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True,
                        help="cpu for Raspberry Pi, cuda for Jetson Nano")
    args = parser.parse_args()

    # rglob finds all .onnx files recursively in all subfolders
    # filters out the _int8_dynamic files, get generated automatically
    # by onnxruntime when running int8 mode and dont want to benchmark them again
    models = sorted([
        p for p in ONNX_DIR.rglob("*.onnx")
        # skips auto-generated quantized copies
        if "_int8_dynamic" not in p.name
    ])

    if not models:
        print(f"[!] no .onnx files found in {ONNX_DIR}")
        sys.exit(1)

    # total number of benchmark runs
    total = len(models) * len(MODES)
    done = 0

    print("=" * 40)
    print(f" EdgePareto ONNX Benchmark Suite")
    print(f" device: {args.device}")
    print(f" found {len(models)} models x {len(MODES)} modes = {total} runs")
    print("=" * 40)

    for model_path in models:
        for mode in MODES:
            done += 1
            print(f"\n[{done}/{total}] {model_path.name} — {mode}")

            # sys.executable makes sure the same venv python is used, not system python
            # check=False means the loop continues even if one benchmark fails
            result = subprocess.run(
                [sys.executable, "-m", "src.scripts.benchmark_onnx",
                 "--model", str(model_path),
                 "--mode", mode,
                 "--device", args.device],
                check=False
            )

            if result.returncode != 0:
                print(f"[!] something went wrong with {model_path.name} {mode} — skipping...")

    print("\n" + "=" * 40)
    print(" all benchmarks complete")
    print(" results saved to models/results/")
    print("=" * 40)
