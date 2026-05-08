import re
import torch
import torch_pruning as tp
import onnxruntime as ort
import src.config as cfg

from pathlib import Path
from src.setup.resnet_setup import get_resnet
from src.utils.utils import setup_reproducibility


# Extracts accuracy value from a weights filename (e.g. "...acc38.84_weights.pth" → "38.84")
def extract_acc(weights_path):
    match = re.search(r"acc([\d]+\.[\d]+)", Path(weights_path).name)
    return match.group(1) if match else "unknown"


# Exports a model to ONNX and verifies it with a dummy forward pass
def export_to_onnx(model, save_path):
    model.eval()
    dummy_input = torch.randn(1, 3, 64, 64)

    torch.onnx.export(
        model,
        dummy_input,
        str(save_path),
        dynamo=False,
        opset_version=cfg.ONNX_OPSET,
        input_names=["input"],
        output_names=["output"]
    )
    print(f"[!] exported: {save_path.name}")

    # verifies the exported model with onnxruntime
    session = ort.InferenceSession(str(save_path))
    outputs = session.run(None, {"input": dummy_input.numpy()})
    print(f"[*] verification passed — output shape: {outputs[0].shape}")


# Loads the FP32 baseline model from WEIGHTS_BASELINE
def load_baseline(device):
    candidates = sorted(cfg.WEIGHTS_BASELINE.glob("best_baseline_acc*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No baseline weights found in {cfg.WEIGHTS_BASELINE}")

    model = get_resnet(num_classes=cfg.NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(torch.load(candidates[-1], map_location=device))
    print(f"[*] baseline loaded: {candidates[-1].name}")
    return model


# Replays the MetaPruner surgery and loads the pruned weights
# The architectural surgery are replayed so layer shapes match the saved state dict
def load_pruned(weights_path, pruning_level, device):
    model = get_resnet(num_classes=cfg.NUM_CLASSES, pretrained=False).to(device)

    example_inputs = torch.rand(1, 3, 64, 64).to(device)
    pruner = tp.pruner.MetaPruner(
        model, example_inputs,
        importance=tp.importance.MagnitudeImportance(p=1),
        pruning_ratio=pruning_level,
        ignored_layers=[model.fc]
    )
    pruner.step()

    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"[*] pruned model loaded: {weights_path.name}")
    return model


if __name__ == "__main__":
    setup_reproducibility(cfg.SEED)
    device = torch.device("cpu")  # ONNX export runs on CPU

    cfg.ONNX_BASELINE.mkdir(parents=True, exist_ok=True)
    cfg.ONNX_STANDALONE.mkdir(parents=True, exist_ok=True)
    cfg.ONNX_HYBRID.mkdir(parents=True, exist_ok=True)

    # 1. baseline (used for FP32 reference and isolated INT8 evaluation on hardware)
    print("\n[*] exporting baseline...")
    try:
        candidates = sorted(cfg.WEIGHTS_BASELINE.glob("best_baseline_acc*.pth"))
        acc = extract_acc(candidates[-1])
        model = load_baseline(device)
        export_to_onnx(model, cfg.ONNX_BASELINE / f"resnet18_baseline_acc{acc}_fp32.onnx")
    except Exception as e:
        print(f"[!] baseline export failed: {e}")

    # 2. standalone pruned models (FP32 — isolates pruning effect on hardware)
    for level in cfg.PRUNING_RATIOS:
        tag = f"p{int(level * 100)}"
        print(f"\n[*] exporting standalone {tag}...")
        candidates = sorted(cfg.WEIGHTS_STANDALONE.glob(f"resnet18_{tag}_finetuned_acc*_weights.pth"))
        if not candidates:
            print(f"[!] no standalone weights found for {tag} — skipping")
            continue
        try:
            acc = extract_acc(candidates[-1])
            model = load_pruned(candidates[-1], level, device)
            export_to_onnx(model, cfg.ONNX_STANDALONE / f"resnet18_{tag}_standalone_finetuned_acc{acc}_fp32.onnx")
        except Exception as e:
            print(f"[!] standalone {tag} export failed: {e}")

    # 3. hybrid pruned models (FP32 — converted to INT8 by hardware runtime)
    for level in cfg.PRUNING_RATIOS:
        tag = f"p{int(level * 100)}"
        print(f"\n[*] exporting hybrid {tag}...")
        candidates = sorted(cfg.WEIGHTS_HYBRID.glob(f"resnet18_{tag}_hybrid_finetuned_acc*_weights.pth"))
        if not candidates:
            print(f"[!] no hybrid weights found for {tag} — skipping")
            continue
        try:
            acc = extract_acc(candidates[-1])
            model = load_pruned(candidates[-1], level, device)
            export_to_onnx(model, cfg.ONNX_HYBRID / f"resnet18_{tag}_hybrid_finetuned_acc{acc}_fp32.onnx")
        except Exception as e:
            print(f"[!] hybrid {tag} export failed: {e}")

    print("\n[*] ONNX export complete.")
