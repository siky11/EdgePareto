import torch
import torch_pruning as tp
import torch.nn as nn
import os

from src.setup.tiny_data_loader import get_tiny_imagenet_loaders
from src.utils.report_generator import generate_report
from src.setup.resnet_setup import get_resnet
from src.utils.utils import setup_reproducibility
from src.utils.evaluation import evaluate_pruning_stage

def apply_pruning(model_path,target_device, pruning_ratio=0.3):

    # 1. model initialization
    model = get_resnet(num_classes=200, pretrained=False)

    # 2. loads trained weights
    model.load_state_dict(torch.load(model_path, map_location=target_device))
    model.to(target_device)
    model.eval()
    print(f"[*] baseline model loaded: {model_path}")

    # 3. dummy input for dependency analysis
    example_inputs = torch.rand(1, 3, 64, 64).to(target_device)

    # 4. pruning methods gets established (magnitude L1.Norm)
    importance = tp.importance.MagnitudeImportance(p=1)

    #ignores classifier fc to ensure 200 class structure
    ignored_layers = [model.fc]

    # metapruner recognizes resnet shortcuts
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=importance,
        pruning_ratio=pruning_ratio,
        ignored_layers=ignored_layers,
    )

    # pruning start
    print(f"[*] starting structured pruning (ratio: {pruning_ratio})...")
    pruner.step()

    print("[!] finished pruning")
    return model


if __name__ == "__main__":
    BASE_MODEL_PATH = "../../models/best_baseline_acc45.78.pth"
    PRUNING_LEVELS = [0.3, 0.5, 0.7]

    # Global hardware initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_reproducibility(seed=42)

    # 1. setup environment once
    train_loader, val_loader = get_tiny_imagenet_loaders(batch_size=32)
    criterion = nn.CrossEntropyLoss()

    for level in PRUNING_LEVELS:
        print(f"\n{'=' * 40}\n[*] Starting Experiment: Pruning Level {level}\n{'=' * 40}")

        # 2. apply the structural change - now passing the global 'device'
        current_model = apply_pruning(BASE_MODEL_PATH, device, pruning_ratio=level)

        # 3. archive the "raw" state
        # passing the same 'device' to ensure consistent benchmarking
        evaluate_pruning_stage(current_model, val_loader, criterion, device, level, "raw")