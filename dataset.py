
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_data(batch_size=64, seed=42):
    transform = transforms.ToTensor()

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    split_generator = torch.Generator().manual_seed(42)
    shuffle_generator = torch.Generator().manual_seed(seed)

    train_data, val_data = random_split(
        training_data,
        [50000, 10000],
        generator=split_generator
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )

    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False
    )

    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train, val, test = get_data()

    print("Training:", len(train.dataset))
    print("Validation:", len(val.dataset))
    print("Testing:", len(test.dataset))
