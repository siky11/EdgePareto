import re
import torch
import torch.nn as nn
import src.config as cfg

from pathlib import Path
from src.setup.tiny_data_loader import get_tiny_imagenet_loaders
from src.utils.utils import setup_reproducibility
from src.utils.evaluation import evaluate_pruning_stage
from src.scripts.prune_baseline import apply_pruning
from src.scripts.fine_tuning_after_pruning import run_recovery_training
from src.scripts.quantization import quantization


def sort_by_acc(candidates):
    def parse_acc(p):
        match = re.search(r"acc(\d+\.\d+)", Path(p).name)
        return float(match.group(1)) if match else 0.0
    return sorted(candidates, key=parse_acc)


if __name__ == "__main__":
    setup_reproducibility(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. setup data once
    train_loader, val_loader = get_tiny_imagenet_loaders(batch_size=cfg.BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()

    # find baseline model
    baseline_candidates = sort_by_acc(cfg.WEIGHTS_BASELINE.glob("best_baseline_acc*.pth"))
    if not baseline_candidates:
        raise FileNotFoundError(f"No baseline model found in {cfg.WEIGHTS_BASELINE}")
    baseline_path = baseline_candidates[-1]
    print(f"[*] using baseline: {baseline_path.name}")

    for level in cfg.PRUNING_RATIOS:
        print(f"\n{'=' * 40}\n[*] Hybrid Pipeline: p{int(level * 100)}\n{'=' * 40}")

        try:
            # 2. structured pruning
            model = apply_pruning(baseline_path, device, pruning_ratio=level)
            evaluate_pruning_stage(model, val_loader, criterion, device, level, "hybrid_raw", workflow="hybrid")

            # 3. fine-tuning to recover accuracy
            run_recovery_training(model, device, train_loader, val_loader, level,
                                  stage_name="hybrid_finetuned", workflow="hybrid")

            # 4. find the saved finetuned weights
            finetuned_candidates = sort_by_acc(cfg.WEIGHTS_HYBRID.glob(f"resnet18_p{int(level * 100)}_hybrid_finetuned_acc*_weights.pth"))
            if not finetuned_candidates:
                print(f"[!] no finetuned weights found for p{int(level * 100)}, skipping quantization...")
                continue
            finetuned_path = finetuned_candidates[-1]
            print(f"[*] using finetuned weights: {finetuned_path.name}")

            # 5. quantize the pruned+finetuned model
            quantization(str(finetuned_path), pruning_level=level,
                         stage_name="hybrid_quantized", workflow="hybrid")

            print(f"[!] hybrid pipeline for p{int(level * 100)} complete.")

        except Exception as e:
            print(f"[!] error for p{int(level * 100)}: {e}")
