import torch
import numpy as np
import random
import os
import platform
import datasets
import json
import time
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

def setup_reproducibility(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #deterministic algorithms are essential for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[*] Global seed set to: {seed}")

def get_software_inventory():
    #gets the name of currently used device
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Only"

    ram_gb = "N/A"
    if psutil is not None:
        # Die IDE weiß jetzt: Hier kann psutil NICHT None sein
        ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    return{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
        "os": platform.platform(),
        "processor": platform.processor(),  # adds CPU info
        "gpu_model": device_name,  # adds GPU name
        "ram_total_gb": ram_gb,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "hf_datasets": datasets.__version__
    }

def save_experiment_log(directory, filename, data):

    # 1. saves timestamp for filename
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    # 2. splits filename and suffix
    name, ext = os.path.splitext(filename)

    # 3. creates new filename with timestamp
    timestamped_filename = f"{name}_{timestamp}{ext}"

    # 4. creates dir if not existent
    os.makedirs(directory, exist_ok=True)

    # 5. builds path and saves file
    path = os.path.join(directory, timestamped_filename)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"[*] Experiment log saved to: {path}")

def get_kernel_characterization(model):
    characterization = {}
    total_flops = 0
    total_params = 0

    # start resolution for tiny imagenet
    current_h, current_w = 64, 64

    for name, module in model.named_modules():
        # A) convolutional layers (kernels)
        if isinstance(module, torch.nn.Conv2d):
            k = module.kernel_size[0]
            s = module.stride[0]
            p = module.padding[0]
            cin = module.in_channels
            cout = module.out_channels

            # saves feature map size for this layer (HW)
            # important that the calculation finishes before the stride reduction for this layer
            layer_hw = [current_h, current_w]

            # calculates output size for next layer
            current_h = (current_h + 2 * p - k) // s + 1
            current_w = (current_w + 2 * p - k) // s + 1

            # FLOPs & params
            layer_flops = 2 * cin * cout * k * k * current_h * current_w
            layer_params = sum(p.numel() for p in module.parameters())

            total_flops += layer_flops
            total_params += layer_params

            characterization[name] = {
                "type": "conv2d",
                "HW": layer_hw,
                "K": [k, k],
                "S": [s, s],
                "P": [p, p],
                "C_in": cin,
                "C_out": cout,
                "flops": layer_flops,
                "params": layer_params
            }

        # B) linear layers (classification head)
        elif isinstance(module, torch.nn.Linear):
            cin = module.in_features
            cout = module.out_features
            layer_flops = 2 * cin * cout
            layer_params = sum(p.numel() for p in module.parameters())

            total_flops += layer_flops
            total_params += layer_params

            characterization[name] = {
                "type": "linear",
                "C_in": cin,
                "C_out": cout,
                "flops": layer_flops,
                "params": layer_params
            }

    return characterization, total_flops, total_params

def measure_90th_latency(model, device, num_samples=500):

    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64).to(device)
    latencies = []

    #Warm-up
    for _ in range(20):
        _ = model(dummy_input)

    with torch.no_grad():
        for _ in range(num_samples):
            start_time = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()    #waits for GPU to finish
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)    #in milliseconds

    latencies.sort()
    p90 = latencies[int(len(latencies) * 0.9)]
    return p90, latencies

def get_model_size_mb(file_path):
    try:
        if os.path.exists(file_path):
            size_bytes = os.path.getsize(file_path)
            return round(size_bytes / (1024 * 1024), 2)
    except OSError:
        pass
    return 0.0

def get_process_memory():
    # Holt die ID des aktuellen Python-Prozesses
    if psutil is not None:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
        return round(mem_mb, 2)
    return 0.0