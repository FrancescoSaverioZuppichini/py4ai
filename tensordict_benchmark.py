import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torchvision.models as models
from torch import autocast, nn, optim
from torch.cuda.amp import GradScaler
from torch.nn.functional import interpolate
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet50, resnet101

from src.dataset import Collate, Data, FolderDataset
from src.utils import get_logger, get_torch_profiler, schedule
from src.vars import BATCH_SIZES
from tqdm import tqdm
import gc

os.environ["TORCH_COMPILE_DEBUG"] = "1"

logger = get_logger()
IMAGE_SIZES = [(640, 480), (1280, 960)]
BATCH_SIZES = [8, 16, 32, 64]
NUM_EPOCHES = 10
DEVICES = ["cpu"]

def run_dataloader(
    ds: Dataset,
    batch_size: int,
    num_epoches: int,
    memmap: bool = False,
    device: str = "cuda",
):
    collate_fn = Collate(device) if memmap else None

    persistent_workers=memmap and device == "cuda"
    print("persiste_worker", persistent_workers)
    dl = DataLoader(
        ds,
        batch_size,
        num_workers=min(batch_size, 8),
        persistent_workers=True,
        pin_memory=False if memmap else True,
        collate_fn=collate_fn,
    )
    # warmup
    print("warmup ...")
    for _ in range(4):
        for batch in tqdm(dl):
            if not memmap and device=="cuda":
                    batch[0] = batch[0].to(device, non_blocking=True)
                    batch[1] = batch[1].to(device, non_blocking=True)
    print("done!")
    with get_torch_profiler() as prof:
        with record_function("run_dataloader"):
            for _ in range(num_epoches):
                for batch in tqdm(dl):
                    if not memmap and device=="cuda":
                        batch[0] = batch[0].to(device, non_blocking=True)
                        batch[1] = batch[1].to(device, non_blocking=True)
    return prof


def do_benchmark():
    models = {"resnet18": resnet18, "resnet50": resnet50}
    datasets_prefix = ["uncompressed" ]
    for device in DEVICES:
        for dataset_prefix in datasets_prefix:
            for img_size in IMAGE_SIZES:
                img_size_str = f"{img_size[0]}_{img_size[1]}"
                ds = FolderDataset(Path(f"data/{dataset_prefix}_{img_size_str}"))
                ds_memmap = Data.from_dataset(Path("tensors") / ds.src.stem, ds)
                for batch_size in BATCH_SIZES:
                    benchmark = f"{dataset_prefix}_{img_size_str}_{batch_size}_{device}"
                    print(f"Running {benchmark}")
                    prof = run_dataloader(
                        ds, batch_size, num_epoches=NUM_EPOCHES, memmap=False, device=device
                    )
                    keys = prof.key_averages()
                    my_keys = list(filter(lambda e: e.key in ["run_dataloader"], keys))
                    for key in my_keys:
                        logger.info(
                            {
                                "key": "memmap",
                                "benchmark": dataset_prefix,
                                "memmap": False,
                                "device": device,
                                "batch_size": batch_size,
                                "img_size": ",".join([str(e) for e in img_size]),
                                "name": key.key,
                                "time": (key.cpu_time_total + key.cuda_time_total)
                                / 1000
                                / NUM_EPOCHES,
                            }
                        )
                    gc.collect()
                    torch.cuda.empty_cache()
                    prof = run_dataloader(
                        ds_memmap, batch_size, num_epoches=NUM_EPOCHES, memmap=True, device=device
                    )
                    keys = prof.key_averages()
                    my_keys = list(filter(lambda e: e.key in ["run_dataloader"], keys))
                    for key in my_keys:
                        logger.info(
                            {
                                "key": "memmap",
                                "benchmark": dataset_prefix,
                                "memmap": True,
                                "device": device,
                                "batch_size": batch_size,
                                "img_size": ",".join([str(e) for e in img_size]),
                                "name": key.key,
                                "time": (key.cpu_time_total + key.cuda_time_total)
                                / 1000
                                / NUM_EPOCHES,
                            }
                        )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    do_benchmark()
#  jq 'select(.message.key == "memmap" and .message.device == "cpu")' logs/my_app.log.jsonl -c |
#  jq '.message' -c  > temp.jsonl && \
# python scripts/make_plots.py --input_file temp.jsonl --aggregation_keys benchmark img_size memmap key name batch_size --group_by_name benchmark --x_axis batch_size --y_axis time --y_axis_lim 0  --name image_to_cpu --output_dir plots