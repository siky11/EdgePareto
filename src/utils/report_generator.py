import torch
from datetime import datetime
from pathlib import Path
from src.utils.utils import (get_software_inventory, get_kernel_characterization,
                   measure_90th_latency, get_model_size_mb, save_experiment_log)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"

def generate_report(model, device, experiment_type, metrics, config, filename_prefix):
    """
    Main reporting methods to aggregate all metrics into a structured JSON format.
    Ensures that both the protocol and the physical weights are preserved.
    """

    print(f"[*] generating comprehensive report for {experiment_type}...")

    # 1. extracts accuracy from metrics
    acc_value = metrics.get("top1_accuracy", 0.0)
    acc_suffix = f"_acc{acc_value:.2f}"

    # 2. prepares path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # filenames
    weights_filename = f"{filename_prefix}{acc_suffix}_weights.pth"
    json_filename = f"{filename_prefix}{acc_suffix}_report.json"

    weights_path = MODELS_DIR / weights_filename
    weights_path_str = str(weights_path)

    # 3. saves phisical weights
    torch.save(model.state_dict(), weights_path_str)
    print(f"[!] weights saved: {weights_filename}")

    # 4. system inventory (SUT digital fingerprint)
    inventory = get_software_inventory()

    # 5. architectural analysis (theoretical complexity)
    arch_summary, total_flops, total_params = get_kernel_characterization(model)

    # 6. performance benchmarking (empirical latency)
    p90_latency, _ = measure_90th_latency(model, device)

    # 7. physical size measurement
    model_size_mb = get_model_size_mb(weights_path_str)

    # 8. data aggregation (single source of truth)
    # combining automated measurements with training metrics
    full_report = {
        "metadata": {
            "experiment_type": experiment_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "associated_weights": weights_filename  # linking the JSON to the .pth file
        },
        "inventory": inventory,
        "architecture_summary": arch_summary,
        "metrics": {
            "top1_accuracy": acc_value,
            "theoretical_GFLOPs": total_flops / 1e9,
            "total_parameters_M": total_params / 1e6,
            "physical_size_mb": model_size_mb,
            "latency_p90_ms": p90_latency,
            "total_training_time_sec": metrics.get("total_training_time_sec", 0.0),
            "final_val_loss": metrics.get("final_val_loss", 0.0),
            "stage": metrics.get("stage", "unknown")
        },
        "config": config
    }

    # 7. save final protocol
    save_experiment_log(str(MODELS_DIR), json_filename, full_report)

    print(f"[!] final report and weights generated for: {filename_prefix}")
    return full_report