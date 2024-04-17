from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def data_loader(
        data_dir,
        batch_size,
        val_size=0.1,
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
    val_split = int(val_size * num_train)
    train_split = num_train - val_split
    train_subset, val_subset = random_split(train_dataset, [train_split, val_split])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size)

    return train_loader, val_loader
