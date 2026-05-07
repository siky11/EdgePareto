import torch
import torch.nn as nn
import torch.ao.quantization as quant
from torchvision.models.quantization import resnet18 as quant_resnet18
from pathlib import Path
import time
import os

from src.setup.tiny_data_loader import get_tiny_imagenet_loaders
from src.utils.utils import setup_reproducibility
from src.utils.evaluation import evaluate_pruning_stage


def quantization(weights_path, pruning_level=0.0):
    setup_reproducibility(seed=42)
    device = torch.device("cpu")

    # 1. Daten laden
    train_loader, val_loader = get_tiny_imagenet_loaders(batch_size=32)
    criterion = nn.CrossEntropyLoss()

    # 2. Architektur-Skelett ohne den fehlerhaften Parameter
    print("[*] Initialisiere quantisierbare ResNet-18 Infrastruktur...")
    # Die Funktion aus dem quantization-Unterordner liefert automatisch das Skelett mit Stubs
    model = quant_resnet18(num_classes=200)

    # 3. Deine Gewichte laden
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Datei nicht gefunden: {weights_path}")

    state_dict = torch.load(weights_path, map_location=device)
    # strict=False ist wichtig, da die Architektur jetzt interne Quantisierungs-Variablen hat
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"[*] Deine Gewichte erfolgreich portiert: {Path(weights_path).name}")

    # 4. Operator Fusion
    # Das offizielle Modell hat diese Methode eingebaut
    model.fuse_model()

    # 5. Quantisierungs-Konfiguration
    # 'fbgemm' für Windows/x86 Simulation
    model.qconfig = quant.get_default_qconfig('fbgemm')

    # 6. Vorbereitung (Observers aktivieren)
    model_prepared = quant.prepare(model)

    # 7. Kalibrierung
    print("[*] Kalibrierung läuft...")
    with torch.no_grad():
        for i, (images, _) in enumerate(train_loader):
            if i >= 64: break
            model_prepared(images)

    # 8. Konvertierung in INT8
    print("[*] Konvertiere Modell in statisches INT8...")
    start_time = time.perf_counter()
    model_int8 = quant.convert(model_prepared)
    process_duration = time.perf_counter() - start_time

    # 9. Finale Evaluation
    evaluate_pruning_stage(
        model=model_int8,
        v_loader=val_loader,
        crit=criterion,
        target_device=device,
        pruning_level=pruning_level,
        stage_name="quantized_final",
        total_time=process_duration,
        final_loss=0.0
    )


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    BASELINE_PATH = base_dir / "models" / "best_baseline_acc45.78.pth"
    quantization(str(BASELINE_PATH), pruning_level=0.0)