import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import get_kernel_characterization, measure_90th_latency
from tiny_data_loader import get_tiny_imagenet_loaders
from resnet_setup import get_resnet
from utils import setup_reproducibility, get_software_inventory, save_experiment_log

#calculates validation metrics to monitor accuracy degradation
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100 * correct / total
    return val_loss / len(val_loader), accuracy


def train_baseline():
    # 1. SUT definition & reproducibility
    setup_reproducibility(seed=42)
    inventory = get_software_inventory()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Setup Data & Model
    # loads tiny according to specified scope
    train_loader, val_loader = get_tiny_imagenet_loaders(batch_size=32)
    model = get_resnet(num_classes=200).to(device)

    # standard cross entropy loss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    epochs = 20  # balanced count to prevent excessive training time

    print(f"[*] starting baseline training on {device}...")

    for epoch in range(epochs):
        model.train()
        # training loop
        train_loss = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # every 500 batches short update
            if (batch_idx + 1) % 500 == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # 3. validation & checkpointing
        # measures top-1 accuracy
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs} - Val Acc: {val_acc:.2f}%")

        #saves optimal FP32-baseline weights for later pruning stages
        if val_acc > best_acc:
            best_acc = val_acc
            model_path = f"../models/best_baseline_acc{best_acc:.2f}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"[!] new best model saved: {model_path}")

    # 4. final baseline characterization
    print("[*] running final baseline characterization...")

    # architecture scan (kernel vectors)
    arch_summary, total_flops, total_params = get_kernel_characterization(model)

    # latency measurement (90th percentile)
    p90_latency, _ = measure_90th_latency(model, device)
    print(f"[*] baseline latency (90th Percentile): {p90_latency:.4f} ms")
    print(f"[*] theoretical complexity: {total_flops / 1e6:.2f} MFLOPs")

    # 5. saves final protocol
    results = {
        "inventory": inventory,
        "architecture_summary": arch_summary,
        "metrics": {
            "top1_accuracy": best_acc,
            "theoretical_GFLOPs": total_flops / 1e9,  # in Giga-FLOPs
            "total_parameters_M": total_params / 1e6,  # in million 
            "latency_p90_ms": p90_latency,
            "final_val_loss": val_loss
        },
        "config": {
            "epochs": epochs,
            "batch_size": 32,
            "optimizer": "Adam",
            "lr": 0.001,
            "seed": 42
        }
    }
    save_experiment_log("../models/", "baseline_fp32_report.json", results)

if __name__ == "__main__":
    train_baseline()