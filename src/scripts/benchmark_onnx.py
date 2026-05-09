import argparse
import time
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
from datetime import datetime


WARMUP_RUNS  = 20
LATENCY_RUNS = 500
OFFLINE_RUNS = 1024


# Builds the ONNX Runtime session based on mode and device
# For INT8 on CPU: quantizes the model dynamically before creating the session
# For INT8 on CUDA: enables TensorRT INT8 execution via TensorrtExecutionProvider
def get_session(model_path, mode, device):
    model_path = Path(model_path)

    if mode == "int8" and device == "cpu":
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantized_path = model_path.parent / f"{model_path.stem}_int8_dynamic.onnx"
        if not quantized_path.exists():
            print("[*] applying dynamic INT8 quantization...")
            quantize_dynamic(str(model_path), str(quantized_path), weight_type=QuantType.QInt8)
        return ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])

    elif mode == "int8" and device == "cuda":
        providers = [
            ("TensorrtExecutionProvider", {"trt_int8_enable": True}),
            "CUDAExecutionProvider"
        ]
        return ort.InferenceSession(str(model_path), providers=providers)

    elif device == "cuda":
        return ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    else:
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


# Single-Stream: one sample at a time, returns P90 latency in ms
# Same logic as measure_90th_latency in utils.py but for ONNX Runtime sessions
def measure_latency_p90(session):
    dummy_input = np.random.randn(1, 3, 64, 64).astype(np.float32)
    input_name = session.get_inputs()[0].name

    for _ in range(WARMUP_RUNS):
        session.run(None, {input_name: dummy_input})

    latencies = []
    for _ in range(LATENCY_RUNS):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()
    p90 = latencies[int(len(latencies) * 0.9)]
    return p90


# Offline: all samples available at once, returns throughput in samples/sec
def measure_throughput(session):
    dummy_input = np.random.randn(1, 3, 64, 64).astype(np.float32)
    input_name = session.get_inputs()[0].name

    for _ in range(WARMUP_RUNS):
        session.run(None, {input_name: dummy_input})

    start = time.perf_counter()
    for _ in range(OFFLINE_RUNS):
        session.run(None, {input_name: dummy_input})
    total_time = time.perf_counter() - start

    return OFFLINE_RUNS / total_time


def main():
    parser = argparse.ArgumentParser(description="ONNX benchmark for edge hardware")
    parser.add_argument("--model",  required=True, help="path to .onnx model file")
    parser.add_argument("--mode",   choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--device", choices=["cpu", "cuda"],  default="cpu")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[*] model:  {model_path.name}")
    print(f"[*] mode:   {args.mode}")
    print(f"[*] device: {args.device}")

    session = get_session(model_path, args.mode, args.device)
    print(f"[*] providers: {session.get_providers()}")

    # single-stream benchmark
    print(f"\n[*] running Single-Stream benchmark ({LATENCY_RUNS} samples)...")
    p90_ms = measure_latency_p90(session)
    print(f"[!] P90 latency: {p90_ms:.3f} ms")

    # offline benchmark
    print(f"[*] running Offline benchmark ({OFFLINE_RUNS} samples)...")
    throughput = measure_throughput(session)
    print(f"[!] throughput: {throughput:.1f} samples/sec")

    # save results
    results_dir = Path(__file__).resolve().parent.parent.parent / "models" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result = {
        "metadata": {
            "timestamp": timestamp,
            "model": model_path.name,
            "mode": args.mode,
            "device": args.device,
            "providers": session.get_providers()
        },
        "single_stream": {
            "p90_latency_ms": round(p90_ms, 4),
            "num_samples": LATENCY_RUNS,
            "warmup_runs": WARMUP_RUNS
        },
        "offline": {
            "throughput_samples_per_sec": round(throughput, 2),
            "num_samples": OFFLINE_RUNS
        }
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = results_dir / f"benchmark_{model_path.stem}_{args.mode}_{args.device}_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"\n[*] results saved: {out_path.name}")


if __name__ == "__main__":
    main()
