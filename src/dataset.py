from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from tensordict import MemoryMappedTensor, tensorclass
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from tqdm import tqdm



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


@tensorclass
class Data:
    images: torch.Tensor
    targets: torch.Tensor

    @classmethod
    def from_dataset(
        cls, out_dir: Path, dataset: FolderDataset, batch_size: int = 64
    ) -> Data:
        out_dir.mkdir(exist_ok=True, parents=True)
        # borrow and adapted from https://pytorch.org/tensordict/tutorials/tensorclass_imagenet.html
        data = cls(
            images=MemoryMappedTensor.empty(
                (
                    len(dataset),
                    *dataset[0][0].squeeze().shape,
                ),
                dtype=torch.uint8,
            ),
            targets=MemoryMappedTensor.empty((len(dataset),), dtype=torch.int64),
            batch_size=[len(dataset)],
        )
        # locks the tensorclass and ensures that is_memmap will return True.
        data.memmap_(str(out_dir))

        dl = DataLoader(dataset, batch_size=batch_size, num_workers=min(batch_size, 8))
        i = 0
        pbar = tqdm(total=len(dataset))
        for image, target in dl:
            _batch = image.shape[0]
            pbar.update(_batch)
            data[i : i + _batch] = cls(
                images=image, targets=target, batch_size=[_batch]
            )
            i += _batch

        return data


class Collate(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device

    def __call__(self, x: Data) -> Data:
        out = x
        if self.device == "cuda":
            out = x.to(self.device, non_blocking=True)
        return out


if __name__ == "__main__":
    # ds = DummyDataset(num_images=10, img_size=(640, 480))
    # print(ds[0])
    ds = FolderDataset(Path("data/compressed_q:v3_640_480"))
    print(ds[0])
    ds = Data.from_dataset(Path("tensors") / ds.src.stem, ds)
    batch_size = 8
    dl = DataLoader(
            ds,
            batch_size,
            num_workers=8,
            # pin_memory=False if memmap else True,
            collate_fn=Collate("cuda"),
        )

    for _ in range(2):
        for batch in dl:
            print(batch.images.device.type)

