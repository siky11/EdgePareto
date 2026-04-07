import torch
import torch_pruning as tp
import os

# Importe aus deinen bestehenden Dateien
from resnet_setup import get_resnet
from utils import setup_reproducibility, measure_90th_latency

def apply_pruning(model_path, pruning_ratio=0.3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_reproducibility(seed=42)

    # 1. model initialization
    model = get_resnet(num_classes=200, pretrained=False)

    # 2. loads trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[*] baseline model loaded: {model_path}")

    # 3. dummy input for dependency analysis
    example_inputs = torch.rand(1, 3, 64, 64).to(device)

    # 4. pruning logic gets established (magnitude L1.Norm)
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
    #path zo model for now hard coded
    BASE_MODEL_PATH = "../models/best_baseline_acc45.78.pth"

    if not os.path.exists(BASE_MODEL_PATH):
        print(f"error: file {BASE_MODEL_PATH} not found!")
    else:
        # execute pruning
        pruned_model = apply_pruning(BASE_MODEL_PATH, pruning_ratio=0.3)

        # verification
        print("\n--- architecture check after pruning ---")

        #check first convulutional layer should now be fewer than 64
        print(f"conv1 output channels: {pruned_model.conv1.out_channels}")

        # test inference
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_inputs = torch.rand(1, 3, 64, 64).to(device)
        with torch.no_grad():
            output = pruned_model(test_inputs)
        print(f"inference test successful! output-shape: {output.shape}")

        # latency check
        p90, _ = measure_90th_latency(pruned_model, device)
        print(f"new p90 latency (pruned, without fine tuning): {p90:.4f} ms")
