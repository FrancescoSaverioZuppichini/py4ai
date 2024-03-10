from torch.utils.data import Dataset
from torchvision.io import read_image
from pathlib import Path
import torch
from typing import Tuple


class FolderDataset(Dataset):
    def __init__(self, src: Path):
        self.src = src
        self.files = list(src.glob("*"))

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        img = read_image(str(self.files[index]))
        label = torch.tensor(1)
        return img, label

    def __len__(self) -> int:
        return len(self.files)


class DummyDataset(Dataset):
    def __init__(self, num_images: int, img_size: Tuple[int, int]):
        self.num_images = num_images
        self.img_size = img_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = (torch.rand((3, *self.img_size)) * 255).to(dtype=torch.uint8)
        label = torch.tensor(1)
        return img, label

    def __len__(self) -> int:
        return self.num_images


if __name__ == "__main__":
    ds = FolderDataset(Path("data/uncompressed"))
    print(ds[0])
    ds = DummyDataset(num_images=10, img_size=(640, 480))
    print(ds[0])
