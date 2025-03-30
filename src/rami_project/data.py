from pathlib import Path

import torch
import typer
from torch.utils.data import Dataset


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def corrupt_mnist() -> tuple[Dataset, Dataset]:

    """Return train and test datasets for corrupt MNIST."""
    processed_path = Path("data/processed")

    train_images = torch.load(processed_path / "train_images.pt")
    train_target = torch.load(processed_path / "train_target.pt")
    test_images = torch.load(processed_path / "test_images.pt")
    test_target = torch.load(processed_path / "test_target.pt")

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set



def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data and save it to processed directory."""
    raw_path = Path(raw_dir)   # ✅ Add "corruptmnist"
    processed_path = Path(processed_dir)

    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(raw_path / f"train_images_{i}.pt"))
        train_target.append(torch.load(raw_path / f"train_target_{i}.pt"))

    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(raw_path / "test_images.pt")
    test_target: torch.Tensor = torch.load(raw_path / "test_target.pt")

    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = (train_images - train_images.mean()) / train_images.std()
    test_images = (test_images - test_images.mean()) / test_images.std()

    processed_path.mkdir(parents=True, exist_ok=True)

    torch.save(train_images, processed_path / "train_images.pt")
    torch.save(train_target, processed_path / "train_target.pt")
    torch.save(test_images, processed_path / "test_images.pt")
    torch.save(test_target, processed_path / "test_target.pt")

if __name__ == "__main__":
    typer.run(preprocess_data)
