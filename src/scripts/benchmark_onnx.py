import argparse
import re
import time
import threading
import platform
import numpy as np
import onnxruntime as ort
from pathlib import Path
from datetime import datetime

# psutil is used for reading CPU%, RAM and temperature during inference
try:
    import psutil
except ImportError:
    psutil = None

from src.utils.utils import get_model_size_mb, save_experiment_log, get_process_memory


# --- benchmark config ---
# warmup runs are important! first few inferences are always slower becuase of
# caching effects, JIT compilation etc. -> throwa them away and doesnt measure them
WARMUP_RUNS  = 20

# samples that are actually measured for latency (Single-Stream scenario)
LATENCY_RUNS = 500

# samples for throughput measurment (Offline scenario )
OFFLINE_RUNS = 1024


# reads the board model from the linux device tree - works on raspberry pi and jetson nano
# on windows this path doesnt exist so we fall back to the hostname
def detect_target_device():
    try:
        model = Path("/proc/device-tree/model").read_text().strip().rstrip("\x00")
        return model
    except (FileNotFoundError, OSError):
        return platform.node() or "Unknown"


# tries to extract pruning_ratio and workflow type directly from the filename
def parse_model_info(model_path):
    name = Path(model_path).stem

    # regex looks for pattern like "_p30_" or "_p70_" in the filename
    ratio_match    = re.search(r"_p(\d+)_", name)

    # checks if the model was trained in hybrid or standalone workflow
    workflow_match = re.search(r"_(hybrid|standalone)_", name)

    # divide by 100 to convert 30 -> 0.3
    pruning_ratio  = int(ratio_match.group(1)) / 100 if ratio_match else 0.0
    workflow       = workflow_match.group(1) if workflow_match else "standalone"

    return pruning_ratio, workflow


# builds the hardware/software inventory for the benchmark report
# important: this is intentionaly different from get_software_inventory() in utils.py
# becuase edge devices dont have PyTorch, CUDA or HuggingFace installed
def get_edge_inventory(session):
    mem = psutil.virtual_memory() if psutil else None
    return {
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        # physical cores matter more than logical ones for edge devices
        "cpu_cores_physical": psutil.cpu_count(logical=False) if psutil else None,
        "ram_total_mb": round(mem.total / (1024 ** 2)) if mem else None,
        "python_version": platform.python_version(),
        # onnx runtime version is teh key dependency here, not pytorch
        "onnxruntime_version": ort.__version__,
        # shows which execution backend is actually being used (CPU, CUDA, TensorRT...)
        "providers_active": session.get_providers(),
    }


# runs in a background thread and continuosly samples hardware metrics
# while the inference benchmark is running in the main thread
# -> this way gets real measurements during actual inference, not before or after
class HardwareMonitor:

    def __init__(self, interval=0.5):
        # sample every 0.5 seconds
        self.interval = interval

        # threading.Event is used to signal the background thread to stop
        self._stop = threading.Event()

        # lists to collect samples over time, averages them later in summary()
        self.cpu_samples = []
        self.ram_samples = []
        self.process_ram_samples = []
        self.temp_samples = []
        self.gpu_samples = []

        # daemon=True thread gets killed automatically when the main program exits
        self._thread = threading.Thread(target=self._run, daemon=True)

        # first call to cpu_percent always returns 0.0 because it needs a
        # reference point -> we call it once here to "prime" it and throw away the result
        if psutil:
            psutil.cpu_percent(interval=None)

    def start(self):
        # clears the stop signal and starts the background thread
        self._stop.clear()
        self._thread.start()

    def stop(self):
        # signals the thread to stop and wait until it fully finishes
        self._stop.set()
        self._thread.join()

    def _run(self):
        # this loop runs in the background thread until stop() is called
        while not self._stop.is_set():
            if psutil:
                # system-wide CPU usage in percent
                self.cpu_samples.append(psutil.cpu_percent(interval=None))

                # total system RAM currently in use (in MB)
                self.ram_samples.append(psutil.virtual_memory().used / (1024 ** 2))

                # process-specific memory footprint - this is what the model actualy needs
                # more precise than system RAM becuase it only counts our python process
                self.process_ram_samples.append(get_process_memory())

                # temperature reading that only works on linux (raspberry pi, jetson)
                # on windows sensors_temperatures() raises NotImplementedError -> we catch it
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        # flatten all sensor readings and take the highest one
                        all_temps = [t.current for readings in temps.values() for t in readings]
                        if all_temps:
                            self.temp_samples.append(max(all_temps))
                except (AttributeError, NotImplementedError):
                    pass

                # GPU utilization via sysfs, works on Jetson Nano only
                # returns 0-1000, divide by 10 to get percentage
                # on raspberry pi and windows this path doesnt exist -> silently skipped
                try:
                    gpu_load = int(Path("/sys/devices/gpu.0/load").read_text().strip())
                    self.gpu_samples.append(round(gpu_load / 10, 1))
                except (FileNotFoundError, ValueError):
                    pass

            # wait for the next interval (or until stop() is called)
            self._stop.wait(self.interval)

    # calculates average/max from all collected samples
    # returns None for metrics where no data was collected (temp on windows)
    def summary(self):
        return {
            "cpu_util_avg_pct": round(float(np.mean(self.cpu_samples)), 1) if self.cpu_samples else None,
            "gpu_util_avg_pct": round(float(np.mean(self.gpu_samples)), 1) if self.gpu_samples else None,
            "ram_used_avg_mb": round(float(np.mean(self.ram_samples)), 1) if self.ram_samples else None,
            "process_ram_avg_mb": round(float(np.mean(self.process_ram_samples)), 1) if self.process_ram_samples else None,
            # max temp is important to detect thermal throttling during the benchmark
            "temp_max_c": round(max(self.temp_samples), 1) if self.temp_samples else None,
            "temp_avg_c": round(float(np.mean(self.temp_samples)), 1) if self.temp_samples else None,
        }


# reads power consumption from the linux sysfs filesystem
# this works on raspberry pi and some jetson configurations
# the value is stored in microwatts divided by 1_000_000 to get watts
# on windows paths dont exist -> returns None
def read_power_watts():
    sysfs_paths = [
        "/sys/class/power_supply/BAT0/power_now",
        "/sys/class/power_supply/BAT1/power_now",
    ]
    for path in sysfs_paths:
        try:
            microwatts = int(Path(path).read_text().strip())
            return round(microwatts / 1_000_000, 3)
        except (FileNotFoundError, ValueError):
            continue
    return None


# creates correct ONNX Runtime session depending on mode and target device
# the session is basically the "loaded model" that runs inference with
def get_session(model_path, mode, device):
    model_path = Path(model_path)

    if mode == "int8" and device == "cpu":
        # dynamic quantization: weights quantized to INT8, activations stay FP32
        # quantizes on the fly if the quantized version doesnt exist yet
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantized_path = model_path.parent / f"{model_path.stem}_int8_dynamic.onnx"
        if not quantized_path.exists():
            print("[*] applying dynamic INT8 quantization...")
            quantize_dynamic(str(model_path), str(quantized_path), weight_type=QuantType.QInt8)
        return ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])

    elif mode == "int8" and device == "cuda":
        # TensorRT can do INT8 inference on NVIDIA GPUs - faster than FP32 on GPU
        providers = [
            ("TensorrtExecutionProvider", {"trt_int8_enable": True}),
            "CUDAExecutionProvider"
        ]
        return ort.InferenceSession(str(model_path), providers=providers)

    elif device == "cuda":
        # both FP32 and INT8 use TensorrtExecutionProvider for a consistent comparison
        # this isolates the quantization effect from the TensorRT graph optimization effect
        # on Jetson Nano (Maxwell) native INT8 is not supported, so both modes compute in FP32
        return ort.InferenceSession(str(model_path), providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])

    else:
        # default: cpu inference in fp32
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


# Single-Stream benchmark -> measures latency for one sample at a time
# also captures hardware metrics (CPU, RAM, temp, power) DURING the inference run
def measure_latency_p90(session):
    # creates random dummy image with the same shape as tiny-imagenet (3x64x64)
    # doesn't need real images here, only measuring speed not accuracy
    dummy_input = np.random.randn(1, 3, 64, 64).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # warmup
    for _ in range(WARMUP_RUNS):
        session.run(None, {input_name: dummy_input})

    # starts hardware monitoring right before the actual measurment begins
    monitor = HardwareMonitor()
    power_before = read_power_watts()
    monitor.start()

    latencies = []
    for _ in range(LATENCY_RUNS):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        # multiply by 1000 to convert seconds to milliseconds
        latencies.append((time.perf_counter() - start) * 1000)

    # samples power again after the run and averages both readings
    power_after = read_power_watts()
    monitor.stop()

    # sort latencies and take the 90th percentile
    # P90 means: 90% of all requests were faster than this value
    # more robust than average because it captures occasional slow spikes
    latencies.sort()
    p90 = latencies[int(len(latencies) * 0.9)]

    hw = monitor.summary()
    # average the two power readings (before and after) as an approximation
    # not perfect but good enough without dedicated power metering hardware
    if power_before is not None and power_after is not None:
        hw["power_avg_w"] = round((power_before + power_after) / 2, 3)
    else:
        hw["power_avg_w"] = None

    return p90, hw


# Offline benchmark: measures throughput when all data is available at once
# result is samples per second (higher = better)
def measure_throughput(session):
    dummy_input = np.random.randn(1, 3, 64, 64).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # warmup again, same reason as above
    for _ in range(WARMUP_RUNS):
        session.run(None, {input_name: dummy_input})

    # measure total time for all offline runs inferences
    start = time.perf_counter()
    for _ in range(OFFLINE_RUNS):
        session.run(None, {input_name: dummy_input})
    total_time = time.perf_counter() - start

    # throughput = samples / total_time
    return OFFLINE_RUNS / total_time


def main():
    # command line arguments - model path is required, rest has defaults
    parser = argparse.ArgumentParser(description="ONNX benchmark for edge hardware")
    parser.add_argument("--model",         required=True, help="path to .onnx model file")
    parser.add_argument("--mode",          choices=["fp32", "int8"], default="fp32")
    parser.add_argument("--device",        choices=["cpu", "cuda"],  default="cpu")
    # these two are optional if not given parses them from the filename automatically
    parser.add_argument("--pruning-ratio", type=float, default=None, help="override pruning ratio (auto-parsed from filename if omitted)")
    parser.add_argument("--workflow",      choices=["standalone", "hybrid"], default=None, help="override workflow (auto-parsed from filename if omitted)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # trys to extract pruning ratio and workflow from the filename
    # CLI args take priority if explicitly provided
    parsed_ratio, parsed_workflow = parse_model_info(model_path)
    pruning_ratio = args.pruning_ratio if args.pruning_ratio is not None else parsed_ratio
    workflow      = args.workflow      if args.workflow      is not None else parsed_workflow

    print(f"[*] model:  {model_path.name}")
    print(f"[*] mode:   {args.mode}")
    print(f"[*] device: {args.device}")

    session = get_session(model_path, args.mode, args.device)
    print(f"[*] providers: {session.get_providers()}")

    # runs single-stream benchmark first, hardware metrics are captured during this run
    print(f"\n[*] running Single-Stream benchmark ({LATENCY_RUNS} samples)...")
    p90_ms, hw_stats = measure_latency_p90(session)
    print(f"[!] P90 latency: {p90_ms:.3f} ms")

    # then offline benchmark for throughput
    print(f"[*] running Offline benchmark ({OFFLINE_RUNS} samples)...")
    throughput = measure_throughput(session)
    print(f"[!] throughput: {throughput:.1f} samples/sec")

    # output goes into models/results/
    results_dir = Path(__file__).resolve().parent.parent.parent / "models" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # assembles the final report, mirrors the training reports for consistency
    result = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_path.name,
            "mode": args.mode,
            "device": args.device,
            "target_device": detect_target_device(),
        },
        # system under test shows exactly which hardware ran the benchmark
        "inventory": get_edge_inventory(session),
        # actual file size on disk important for storage-constrained edge devices
        "storage": {
            "model_size_mb": get_model_size_mb(str(model_path))
        },
        # hardware metrics captured during inference (temp/power = null on windows)
        "hardware": hw_stats,
        # MLPerf Single-Stream scenario result
        "single_stream": {
            "p90_latency_ms": round(p90_ms, 4),
            "num_samples": LATENCY_RUNS,
            "warmup_runs": WARMUP_RUNS,
        },
        # MLPerf Offline scenario result
        "offline": {
            "throughput_samples_per_sec": round(throughput, 2),
            "num_samples": OFFLINE_RUNS
        },
        # experiment config for reproducibility
        "config": {
            "pruning_ratio": pruning_ratio,
            "workflow": workflow,
            "mode": args.mode
        }
    }

    base_filename = f"benchmark_{model_path.stem}_{args.mode}_{args.device}.json"
    save_experiment_log(str(results_dir), base_filename, result)


if __name__ == "__main__":
    main()
