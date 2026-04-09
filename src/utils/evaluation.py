import torch
from src.utils.utils import validate
from src.utils.report_generator import generate_report

#universal evaluation function (Baseline, Raw, Finetuned) ensures consistent reporting
def evaluate_pruning_stage(model, v_loader, crit, target_device, pruning_level, stage_name, total_time, final_loss):

    print(f"\n[*] --- EVALUATION STAGE: {stage_name.upper()} (Ratio: {pruning_level}) ---")

    # 1. accuracy check
    _, acc = validate(model, v_loader, crit, target_device)
    print(f"[!] {stage_name} accuracy: {acc:.2f}%")

    # 2. prepares metrics for report
    metrics = {
        "top1_accuracy": acc,
        "stage": stage_name,
        "total_training_time_sec": round(total_time, 2),
        "final_val_loss": final_loss
    }

    config = {
        "pruning_ratio": pruning_level,
        "criterion": "L1-Norm",
        "stage": stage_name
    }

    # 3. generates official report
    generate_report(
        model=model,
        device=target_device,
        experiment_type=f"Structured Pruning ({stage_name})",
        metrics=metrics,
        config=config,
        filename_prefix=f"resnet18_p{int(pruning_level * 100)}_{stage_name}"
    )

    return acc