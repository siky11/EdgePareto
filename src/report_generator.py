import torch
import os
from datetime import datetime
from utils import (get_software_inventory, get_kernel_characterization,
                   measure_90th_latency, get_model_size_mb, save_experiment_log, validate)


def generate_report(model, device, experiment_type, metrics, config, filename_prefix):
    """
    Main reporting logic to aggregate all metrics into a structured JSON format.
    Ensures that both the protocol and the physical weights are preserved.
    """

    print(f"[*] generating comprehensive report for {experiment_type}...")

    # 1. physical weight storage (permanent save)
    # ensuring we keep the weights for deployment on edge hardware later
    os.makedirs("../models/", exist_ok=True)
    weights_filename = f"{filename_prefix}_weights.pth"
    weights_path = os.path.join("../models/", weights_filename)

    torch.save(model.state_dict(), weights_path)
    print(f"[!] weights saved permanently to: {weights_path}")

    # 2. system inventory (SUT digital fingerprint)
    inventory = get_software_inventory()

    # 3. architectural analysis (theoretical complexity)
    arch_summary, total_flops, total_params = get_kernel_characterization(model)

    # 4. performance benchmarking (empirical latency)
    p90_latency, _ = measure_90th_latency(model, device)

    # 5. physical size measurement
    model_size_mb = get_model_size_mb(weights_path)

    # 6. data aggregation (single source of truth)
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
            "top1_accuracy": metrics.get("top1_accuracy", 0.0),
            "theoretical_GFLOPs": total_flops / 1e9,
            "total_parameters_M": total_params / 1e6,
            "physical_size_mb": model_size_mb,
            "latency_p90_ms": p90_latency,
            **metrics  # includes all additional experimental data
        },
        "config": config
    }

    # 7. save final protocol
    json_filename = f"{filename_prefix}_report.json"
    save_experiment_log("../models/", json_filename, full_report)

    print(f"[!] final report and weights generated for: {filename_prefix}")
    return full_report