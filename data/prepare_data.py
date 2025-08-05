import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import os
import random

def get_cifar100_dataloaders(data_dir="/home/mahyad/projects/ViT_vs_ResNet-50/data", batch_size=64, val_split=0.1, subset_ratio=1.0, num_workers=2):
    # Mean and std of CIFAR-100 for normalization
    mean = (0.5071, 0.4865, 0.4409)
    std = (0.2673, 0.2564, 0.2761)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Load datasets
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

    # Subset train dataset if needed
    if subset_ratio < 1.0:
        indices = random.sample(range(len(train_dataset)), int(len(train_dataset) * subset_ratio))
        train_dataset = Subset(train_dataset, indices)

    # Split train into train/val
    val_size = int(val_split * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_cifar100_dataloaders(subset_ratio=0.5)
    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))
