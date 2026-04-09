import torch
import torch.nn as nn
from torchvision import models

#loads ResNet-18 and customizes settings to required image size
def get_resnet(num_classes=200, pretrained=True):

    # 1. load model
    if pretrained:
        model = models.resnet18(weights='DEFAULT')
        print("pretrained weights loaded.")
    else:
        model = models.resnet18(weights=None)
        print("model initialized without prior knowledge.")

    # 2. customizes last layer(Fully Connected - 'fc')
    in_features = model.fc.in_features

    # 1000 exits get replaced by the 200 classes
    model.fc = nn.Linear(in_features, num_classes)

    return model

if __name__ == "__main__":

    resnet = get_resnet(num_classes=200, pretrained=True)

    #test input
    test_input = torch.randn(32, 3, 64, 64)
    output = resnet(test_input)

    print(f"model-check: input {test_input.shape}, output {output.shape}")


