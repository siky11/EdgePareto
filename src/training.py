import torch
import torch.nn as nn
import torch.optim as optim
from tiny_data_loader import get_tiny_imagenet_loaders
from resnet_setup import get_resnet

def train_one_epoch():
    device = torch.device("cpu")

    # 1. get data and model
    train_loader, val_loader = get_tiny_imagenet_loaders(batch_size=32)
    model = get_resnet(num_classes=200).to(device)

    # 2. loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    print("small training for 10 batches started...")

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 10:
            break

        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f"batch {batch_idx+1}/10 - loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_one_epoch()