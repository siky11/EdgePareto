import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import load_dataset

#Loads tiny-imagenet from hugging face
# creates PyTorch DataLoader, preprocesses and automated downloading
def get_tiny_imagenet_loaders(batch_size=64, cache_dir="../data/hf_cache"):

    print(f"load dataset (cache: {cache_dir})...")

    # 1. loads dataset from hugging face
    dataset = load_dataset("zh-plus/tiny-imagenet", cache_dir=cache_dir)

    # 2. defines transformations
    # normalization values for ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # because ResNet-18 pre-trained on it
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], #mean of color channels
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        #convertion to RGB, because tiny has some grey-step pictures
        transforms.Lambda(lambda x: x.convert("RGB")),  # Wichtig: Graustufen-Bilder zu RGB wandeln

        #this is for overfitting - ensures that classes are still classifiable even if it is rotated
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),

        #converts size to 64x64 pixels - native tiny-image size
        transforms.Resize(64),

        #convertion to tensor
        transforms.ToTensor(),
        normalize
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.Resize(64),
        transforms.ToTensor(),
        normalize
    ])

    # 3. mapping function for transformations
    def apply_train_transforms(examples):
        examples["pixel_values"] = [train_transform(image) for image in examples["image"]]
        return examples

    def apply_val_transforms(examples):
        examples["pixel_values"] = [val_transform(image) for image in examples["image"]]
        return examples

    # assigns transformations on the fly
    # images only transformed when model requests it - more memory efficient
    train_ds = dataset["train"].with_transform(apply_train_transforms)
    val_ds = dataset["valid"].with_transform(apply_val_transforms)

    # 4. collate function, converts HF-Dictionary into PyTorch-Tensors
    def collate_fn(examples):
        # stacks all single images into a 4D-Tensor(matrix) (batch, channel, height and width)
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        #creates tensor from label
        labels = torch.tensor([example["label"] for example in examples])
        return pixel_values, labels

    # 5. creation of loader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # on Windows often more stable, maybe putting it higher on edge hardware
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0
    )

    return train_loader, val_loader

if __name__ == "__main__":
    try:
        t_loader, v_loader = get_tiny_imagenet_loaders(batch_size=32)
        images, labels = next(iter(t_loader))
        print(f"success! batch-size: {images.shape}")
        print(f"number of classes in batch: {len(torch.unique(labels))}")
    except Exception as e:
        print(f"error while loading: {e}")