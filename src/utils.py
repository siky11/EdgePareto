import torch
import numpy as np
import random
import os
import platform
import datasets
import json
from datetime import datetime

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

    return{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
        "os": platform.platform(),
        "processor": platform.processor(),  # adds CPU info
        "gpu_model": device_name,  # adds GPU name
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