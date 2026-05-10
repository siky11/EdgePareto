from pathlib import Path

# Projekt-Wurzelpfade
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_CACHE_DIR = ROOT_DIR / "data" / "hf_cache"

# weights and reports split by workflow
WEIGHTS_DIR = MODELS_DIR / "weights"
REPORTS_DIR = MODELS_DIR / "reports"

WEIGHTS_BASELINE   = WEIGHTS_DIR / "baseline"
WEIGHTS_STANDALONE = WEIGHTS_DIR / "standalone"
WEIGHTS_HYBRID     = WEIGHTS_DIR / "hybrid"

REPORTS_BASELINE   = REPORTS_DIR / "baseline"
REPORTS_STANDALONE = REPORTS_DIR / "standalone"
REPORTS_HYBRID     = REPORTS_DIR / "hybrid"

# Allgemeine Experiment-Einstellungen
SEED = 42
NUM_CLASSES = 200
BATCH_SIZE = 32

# Pruning
PRUNING_RATIOS = [0.3, 0.5, 0.7]

# Training & Fine-Tuning
TRAINING_EPOCHS = 30
FINETUNE_EPOCHS = 15
LR_BASELINE = 1e-3
LR_FINETUNE = 1e-4

# Quantisierung — "qnnpack" für ARM Edge-Hardware
QUANTIZATION_BACKEND = "fbgemm"
CALIBRATION_BATCHES = 64

# ONNX Export
ONNX_DIR        = MODELS_DIR / "onnx"
ONNX_BASELINE   = ONNX_DIR / "baseline"
ONNX_STANDALONE = ONNX_DIR / "standalone"
ONNX_HYBRID     = ONNX_DIR / "hybrid"
ONNX_OPSET      = 18
