import torch
import torch.nn as nn
import torch.optim as optim
import torch_pruning as tp
import os
import time

from src.setup.tiny_data_loader import get_tiny_imagenet_loaders
from src.utils.utils import validate, setup_reproducibility
from src.setup.resnet_setup import get_resnet
from src.utils.evaluation import evaluate_pruning_stage
from pathlib import Path

def load_pruned_model(raw_weights_path, target_device, target_ratio):

    # 1. starts with standard architecture
    model = get_resnet(num_classes=200).to(target_device)

    # 2. applys structural surgery to match the saved state
    # This is required so the layer shapes match the .pth file
    example_inputs = torch.rand(1, 3, 64, 64).to(target_device)
    importance = tp.importance.MagnitudeImportance(p=1)
    ignored_layers = [model.fc]

    pruner = tp.pruner.MetaPruner(
        model, example_inputs, importance=importance,
        pruning_ratio=target_ratio, ignored_layers=ignored_layers
    )
    pruner.step()  # The model now has the reduced dimensions

    # 3. Load the specific weights from your previous pruning run
    if not os.path.exists(raw_weights_path):
        raise FileNotFoundError(f"Raw weights not found: {raw_weights_path}")

    model.load_state_dict(torch.load(raw_weights_path, map_location=target_device))
    print(f"[*] Successfully loaded raw pruned weights: {raw_weights_path}")
    return model


def run_recovery_training(model, target_device, t_loader, v_loader, level, epochs=5):
    print(f"[*] starting iterative retraining for level {level}...")

    start_time = time.time()
    # Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()

    #Adam Optimizer with LR=1e-4
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_recovery_acc = 0.0
    final_loss = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in t_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        #Validation
        final_loss = running_loss / len(t_loader)
        _, current_acc = validate(model, v_loader, criterion, target_device)
        print(f"epoch {epoch+1}/{epochs}, current recovery accuracy {current_acc:.2f}%")

        if current_acc > best_recovery_acc:
            best_recovery_acc = current_acc

    total_time = time.time() - start_time

    evaluate_pruning_stage(
        model=model,
        v_loader=v_loader,
        crit=criterion,
        target_device=target_device,
        pruning_level=level,
        stage_name="finetuned",
        total_time = total_time,
        final_loss=final_loss
    )

    return model

if __name__ == "__main__":

    #level to load
    LEVEL = 0.3
    EPOCHS = 5

    base_dir = Path(__file__).resolve().parent.parent.parent
    RAW_PATH = base_dir / "models" / f"resnet18_p{int(LEVEL * 100)}_raw_weights.pth"

    RAW_PATH_STR = str(RAW_PATH)

    setup_reproducibility(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. setup data
    train_loader, val_loader = get_tiny_imagenet_loaders(batch_size=32)

    # 2. reconstruct and load
    try:
        pruned_model = load_pruned_model(RAW_PATH, device, LEVEL)

        # 3. start fine-tuning
        run_recovery_training(pruned_model, device, train_loader, val_loader, LEVEL, epochs=EPOCHS)
        print(f"\n[!] Recovery for p{int(LEVEL * 100)} finished. Report generated.")

    except Exception as e:
        print(f"[!] Error: {e}")