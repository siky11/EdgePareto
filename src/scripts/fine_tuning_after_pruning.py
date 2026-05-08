import torch
import torch.nn as nn
import torch.optim as optim
import torch_pruning as tp
import time
import src.config as cfg

from src.setup.tiny_data_loader import get_tiny_imagenet_loaders
from src.setup.resnet_setup import get_resnet
from src.utils.utils import validate, setup_reproducibility
from src.utils.evaluation import evaluate_pruning_stage


#Reconstruct architecture and load weights (step must be replayed so shapes match)
# The structural surgery (pruner.step) must be replayed so layer shapes match the .pth file
def load_pruned_model(raw_weights_path, target_device, target_ratio):

    # 1. starts with standard architecture
    model = get_resnet(num_classes=cfg.NUM_CLASSES).to(target_device)

    # 2. replays structural surgery to match the saved state
    example_inputs = torch.rand(1, 3, 64, 64).to(target_device)
    importance = tp.importance.MagnitudeImportance(p=1)
    ignored_layers = [model.fc]

    pruner = tp.pruner.MetaPruner(
        model, example_inputs, importance=importance,
        pruning_ratio=target_ratio, ignored_layers=ignored_layers
    )
    pruner.step()  # model now has the reduced dimensions

    # 3. load the specific weights from the previous pruning run
    if not raw_weights_path.exists():
        raise FileNotFoundError(f"Raw weights not found: {raw_weights_path}")

    model.load_state_dict(torch.load(raw_weights_path, map_location=target_device))
    print(f"[*] successfully loaded raw pruned weights: {raw_weights_path.name}")
    return model


# Fine-tunes the pruned model to recover accuracy
# Runs for a fixed number of epochs and saves a report at the end
def run_recovery_training(model, target_device, t_loader, v_loader, level,
                          epochs=cfg.FINETUNE_EPOCHS, lr=cfg.LR_FINETUNE,
                          stage_name="finetuned", workflow="standalone"):
    print(f"[*] starting iterative retraining for level {level}...")

    start_time = time.time()

    # Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer with LR from config
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_recovery_acc = 0.0
    final_loss = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in t_loader:
            images = images.to(target_device)
            labels = labels.to(target_device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # validation
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
        stage_name=stage_name,
        total_time=total_time,
        final_loss=final_loss,
        workflow=workflow
    )

    return model


if __name__ == "__main__":
    setup_reproducibility(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. setup data once
    train_loader, val_loader = get_tiny_imagenet_loaders(batch_size=cfg.BATCH_SIZE)

    for level in cfg.PRUNING_RATIOS:
        print(f"\n{'=' * 40}\n[*] Fine-tuning p{int(level * 100)}\n{'=' * 40}")

        # find matching raw weights for this level
        raw_candidates = sorted(cfg.WEIGHTS_STANDALONE.glob(f"resnet18_p{int(level * 100)}_raw_acc*_weights.pth"))
        if not raw_candidates:
            raise FileNotFoundError(f"No raw weights found for p{int(level * 100)} in {cfg.WEIGHTS_STANDALONE}")
        raw_path = raw_candidates[0]
        print(f"[*] using raw weights: {raw_path.name}")

        try:
            # 2. reconstruct and load
            pruned_model = load_pruned_model(raw_path, device, level)

            # 3. start fine-tuning
            run_recovery_training(pruned_model, device, train_loader, val_loader, level)
            print(f"[!] recovery for p{int(level * 100)} finished. report generated.")

        except Exception as e:
            print(f"[!] error for p{int(level * 100)}: {e}")
