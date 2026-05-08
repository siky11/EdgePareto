import torch
import src.config as cfg
from src.utils.utils import validate
from src.utils.report_generator import generate_report

#universal evaluation function (Baseline, Raw, Finetuned) ensures consistent reporting
def evaluate_pruning_stage(model, v_loader, crit, target_device, pruning_level, stage_name, total_time=0.0, final_loss=0.0, workflow="standalone", fp32_stats=None):

    print(f"\n[*] --- EVALUATION STAGE: {stage_name.upper()} (Ratio: {pruning_level}) ---")

    # 1. accuracy check
    _, acc = validate(model, v_loader, crit, target_device)
    print(f"[!] {stage_name} accuracy: {acc:.2f}%")

    # 2. determine output dirs based on workflow
    if workflow == "hybrid":
        weights_dir = cfg.WEIGHTS_HYBRID
        reports_dir = cfg.REPORTS_HYBRID
    else:
        weights_dir = cfg.WEIGHTS_STANDALONE
        reports_dir = cfg.REPORTS_STANDALONE

    # 3. prepares metrics for report
    metrics = {
        "top1_accuracy": acc,
        "stage": stage_name,
        "total_training_time_sec": round(total_time, 2),
        "final_val_loss": final_loss
    }

    # pass FP32 arch stats if provided (used as fallback for quantized models)
    if fp32_stats:
        metrics.update(fp32_stats)

    config = {
        "pruning_ratio": pruning_level,
        "workflow": workflow,
        "criterion": "L1-Norm",
        "stage": stage_name
    }

    # 4. generates official report
    generate_report(
        model=model,
        device=target_device,
        experiment_type=f"Structured Pruning ({stage_name})",
        metrics=metrics,
        config=config,
        filename_prefix=f"resnet18_p{int(pruning_level * 100)}_{stage_name}",
        weights_dir=weights_dir,
        reports_dir=reports_dir
    )

    return acc
