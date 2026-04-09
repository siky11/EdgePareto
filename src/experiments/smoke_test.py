import torch
import torchvision
import onnx
import sys

def check_setup():
    print(f"--- EdgePareto Environment Check ---")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Torchvision Version: {torchvision.__version__}")
    print(f"ONNX Version: {onnx.__version__}")

    # Checks if an npu is connected
    cuda_available = torch.cuda.is_available()
    print(f"CUDA (GPU) available: {cuda_available}")
    if cuda_available:
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"------------------------------------")

if __name__ == "__main__":
    check_setup()