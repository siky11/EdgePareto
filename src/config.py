from pathlib import Path

# Projekt-Wurzelpfade
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_CACHE_DIR = ROOT_DIR / "data" / "hf_cache"

# Allgemeine Experiment-Einstellungen
SEED = 42
NUM_CLASSES = 200
BATCH_SIZE = 32

# Pruning
PRUNING_RATIOS = [0.3, 0.5, 0.7]

# Training & Fine-Tuning
TRAINING_EPOCHS = 20
FINETUNE_EPOCHS = 5
LR_BASELINE = 1e-3
LR_FINETUNE = 1e-4

# Quantisierung — "qnnpack" für ARM Edge-Hardware
QUANTIZATION_BACKEND = "fbgemm"
CALIBRATION_BATCHES = 64
