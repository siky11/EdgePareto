import torch
from tiny_data_loader import get_tiny_imagenet_loaders
from resnet_setup import get_resnet

def pipeline_test():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 1. loads data - batch size only 16 for quick test
    _, val_loader = get_tiny_imagenet_loaders(batch_size=16)

    # 2. loads model
    model = get_resnet(num_classes=200).to(device)
    model.eval()    #test mode - disables dropout etc.

    # 3. here images and classes get unpacked separately
    images, labels = next(iter(val_loader))
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    print("\n--- pipeline check results ---")
    print(f"image-batch processed successfully: {images.shape}")
    print(f"prediction generated for classes: {predicted.tolist()}")
    print(f"real classes (labels):     {labels.tolist()}")

    #model still not trained on tiny so accuracy very low and matches will only be random
    correct = (predicted == labels).sum().item()
    print(f"random matches in this batch: {correct}/{labels.size(0)}")
    print("---------------------------------")

if __name__ == "__main__":

    pipeline_test()