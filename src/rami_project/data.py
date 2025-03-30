# from __future__ import annotations

# import matplotlib.pyplot as plt  # only needed for plotting
# import torch
# from mpl_toolkits.axes_grid1 import ImageGrid  # only needed for plotting

# DATA_PATH = "data/rami_project/raw"


# def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
#     """Return train and test dataloaders for corrupt MNIST."""
#     train_images, train_target = [], []
#     for i in range(6):
#         train_images.append(torch.load(f"{DATA_PATH}/train_images_{i}.pt"))
#         train_target.append(torch.load(f"{DATA_PATH}/train_target_{i}.pt"))
#     train_images = torch.cat(train_images)
#     train_target = torch.cat(train_target)

#     test_images: torch.Tensor = torch.load(f"{DATA_PATH}/test_images.pt")
#     test_target: torch.Tensor = torch.load(f"{DATA_PATH}/test_target.pt")

#     train_images = train_images.unsqueeze(1).float()
#     test_images = test_images.unsqueeze(1).float()
#     train_target = train_target.long()
#     test_target = test_target.long()

#     train_set = torch.utils.data.TensorDataset(train_images, train_target)
#     test_set = torch.utils.data.TensorDataset(test_images, test_target)

#     return train_set, test_set


# def show_image_and_target(images: torch.Tensor, target: torch.Tensor) -> None:
#     """Plot images and their labels in a grid."""
#     row_col = int(len(images) ** 0.5)
#     fig = plt.figure(figsize=(10.0, 10.0))
#     grid = ImageGrid(fig, 111, nrows_ncols=(row_col, row_col), axes_pad=0.3)
#     for ax, im, label in zip(grid, images, target):
#         ax.imshow(im.squeeze(), cmap="gray")
#         ax.set_title(f"Label: {label.item()}")
#         ax.axis("off")
#     plt.show()


# if __name__ == "__main__":
#     train_set, test_set = corrupt_mnist()
#     print(f"Size of training set: {len(train_set)}")
#     print(f"Size of test set: {len(test_set)}")
#     print(f"Shape of a training point {(train_set[0][0].shape, train_set[0][1].shape)}")
#     print(f"Shape of a test point {(test_set[0][0].shape, test_set[0][1].shape)}")
#     show_image_and_target(train_set.tensors[0][:25], train_set.tensors[1][:25])



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
