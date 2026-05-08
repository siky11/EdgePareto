import torch
import torch.nn as nn
import torch.ao.quantization as quant
import torch_pruning as tp
from torchvision.models.quantization import resnet18 as quant_resnet18
from pathlib import Path
import time
import src.config as cfg

from src.setup.tiny_data_loader import get_tiny_imagenet_loaders
from src.utils.utils import setup_reproducibility, get_kernel_characterization
from src.utils.evaluation import evaluate_pruning_stage


# Runs static INT8 PTQ on a given model (baseline or pruned+finetuned)
# Quantization always runs on CPU — required by fbgemm/qnnpack backends
def quantization(weights_path, pruning_level=0.0, stage_name="quantized_final", workflow="standalone"):
    device = torch.device("cpu")

    # 1. load data
    train_loader, val_loader = get_tiny_imagenet_loaders(batch_size=cfg.BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()

    # 2. quantizable resnet skeleton (has fuse_model() and quant stubs built in)
    print("[*] initializing quantizable ResNet-18 skeleton...")
    model = quant_resnet18(num_classes=cfg.NUM_CLASSES)

    # 3. if pruned model: replay structural surgery on the quant skeleton
    # shapes must match before loading the pruned state dict
    if pruning_level > 0.0:
        example_inputs = torch.rand(1, 3, 64, 64)
        pruner = tp.pruner.MetaPruner(
            model, example_inputs,
            importance=tp.importance.MagnitudeImportance(p=1),
            pruning_ratio=pruning_level,
            ignored_layers=[model.fc]
        )
        pruner.step()

    # 4. load weights
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)
    # strict=False needed because the quant skeleton has extra internal variables
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"[*] weights loaded: {Path(weights_path).name}")

    # 5. capture FP32 arch stats before fusion — needed for quantized report
    fp32_arch, fp32_flops, fp32_params = get_kernel_characterization(model)

    # 6. operator fusion
    model.fuse_model()

    # 7. quantization config — switch to 'qnnpack' in cfg for ARM edge deployment
    model.qconfig = quant.get_default_qconfig(cfg.QUANTIZATION_BACKEND)

    # 8. prepare (activates observers)
    model_prepared = quant.prepare(model)

    # 9. calibration
    print("[*] running calibration...")
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if i >= cfg.CALIBRATION_BATCHES:
                break
            model_prepared(images)

    # 10. convert to INT8
    print("[*] converting to static INT8...")
    start_time = time.perf_counter()
    model_int8 = quant.convert(model_prepared)
    process_duration = time.perf_counter() - start_time

    # 11. final evaluation
    evaluate_pruning_stage(
        model=model_int8,
        v_loader=val_loader,
        crit=criterion,
        target_device=device,
        pruning_level=pruning_level,
        stage_name=stage_name,
        total_time=process_duration,
        final_loss=0.0,
        workflow=workflow,
        fp32_stats={"fp32_arch": fp32_arch, "fp32_flops": fp32_flops, "fp32_params": fp32_params}
    )


if __name__ == "__main__":
    setup_reproducibility(cfg.SEED)

    # find the latest baseline model
    baseline_candidates = sorted(cfg.WEIGHTS_BASELINE.glob("best_baseline_acc*.pth"))
    if not baseline_candidates:
        raise FileNotFoundError(f"No baseline model found in {cfg.MODELS_DIR}")
    baseline_path = baseline_candidates[-1]
    print(f"[*] quantizing baseline: {baseline_path.name}")
    quantization(str(baseline_path), pruning_level=0.0)
