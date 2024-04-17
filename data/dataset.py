from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import torch

def data_loader(
        data_dir,
        batch_size,
        val_ratio=0.1,
        subset_ratio=1,
        test=False
):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if test:
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False,
            download=True, transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return test_loader

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=transform
    )

    num_train = len(train_dataset)
    num_train_sub = int(subset_ratio * num_train)
    indices = torch.randperm(num_train)[:num_train_sub]
    train_subset = Subset(train_dataset, indices)

    val_split = int(val_ratio * num_train_sub)
    train_split = num_train_sub - val_split
    train_split_subset, val_split_subset = random_split(train_subset, [train_split, val_split])

    train_loader = DataLoader(train_split_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_split_subset, batch_size=batch_size)

    return train_loader, val_loader
