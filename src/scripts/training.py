import torch
import torch.nn as nn
import torch.optim as optim
import time
from datetime import datetime

from src.config import SEED, NUM_CLASSES, BATCH_SIZE, TRAINING_EPOCHS, LR_BASELINE, WEIGHTS_BASELINE, REPORTS_BASELINE
from src.setup.tiny_data_loader import get_tiny_imagenet_loaders
from src.setup.resnet_setup import get_resnet
from src.utils.utils import (setup_reproducibility, get_software_inventory,
                              save_experiment_log, get_kernel_characterization,
                              measure_90th_latency, get_model_size_mb, validate)

def train_baseline():

    start_time_total = time.time()
    WEIGHTS_BASELINE.mkdir(parents=True, exist_ok=True)
    REPORTS_BASELINE.mkdir(parents=True, exist_ok=True)

    # 1. reproducibility & system info
    setup_reproducibility(seed=SEED)
    inventory = get_software_inventory()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. setup data & model
    train_loader, val_loader = get_tiny_imagenet_loaders(batch_size=BATCH_SIZE)
    model = get_resnet(num_classes=NUM_CLASSES).to(device)

    # standard cross entropy loss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR_BASELINE)

    best_acc = 0.0
    print(f"[*] starting baseline training on {device}...")

    # training loop
    for epoch in range(TRAINING_EPOCHS):
        model.train()
        train_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # short status update every 500 batches
            if (batch_idx + 1) % 500 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # 3. validation & checkpointing
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{TRAINING_EPOCHS} - Val Acc: {val_acc:.2f}%")

        # saves best FP32 weights for later pruning stages
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = WEIGHTS_BASELINE / f"best_baseline_acc{best_acc:.2f}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"[!] new best model saved: {model_path}")

    total_time = time.time() - start_time_total
    print(f"[*] training finished in {total_time / 60:.2f} minutes")

    # 4. reload best weights before characterization
    # (model currently holds last epoch, not best)
    best_model_path = WEIGHTS_BASELINE / f"best_baseline_acc{best_acc:.2f}.pth"
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # 5. final characterization
    print("[*] running final baseline characterization...")
    arch_summary, total_flops, total_params = get_kernel_characterization(model)
    p90_latency, _ = measure_90th_latency(model, device)
    model_size_mb = get_model_size_mb(best_model_path)
    print(f"[*] baseline latency (90th percentile): {p90_latency:.4f} ms")

    # 6. save final report
    results = {
        "metadata": {
            "experiment_type": "Baseline Training (FP32)",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "associated_weights": f"best_baseline_acc{best_acc:.2f}.pth"
        },
        "inventory": inventory,
        "architecture_summary": arch_summary,
        "metrics": {
            "top1_accuracy": best_acc,
            "theoretical_GFLOPs": total_flops / 1e9,
            "total_parameters_M": total_params / 1e6,
            "physical_size_mb": model_size_mb,
            "latency_p90_ms": p90_latency,
            "total_training_time_sec": round(total_time, 2),
            "final_val_loss": val_loss,
            "stage": "baseline"
        },
        "config": {
            "pruning_ratio": 0.0,
            "workflow": "standalone",
            "epochs": TRAINING_EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "lr": LR_BASELINE,
            "seed": SEED
        }
    }
    save_experiment_log(str(REPORTS_BASELINE), "baseline_fp32_report.json", results)

if __name__ == "__main__":
    train_baseline()
