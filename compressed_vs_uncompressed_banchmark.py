from typing import Tuple
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import os
from src.utils import get_logger, get_torch_profiler, schedule
from src.vars import BATCH_SIZES
from torch.utils.data import DataLoader, Dataset
from src.dataset import FolderDataset
from pathlib import Path
from torch import nn
from torch import optim, autocast
from torch.cuda.amp import GradScaler
from torchvision.models import resnet101, resnet50, resnet18
from torch.nn.functional import interpolate

os.environ["TORCH_COMPILE_DEBUG"] = "1"

logger = get_logger()
IMAGE_SIZES = [(640, 480), (1280, 960)]
BATCH_SIZES = [8, 16, 32, 64]
NUM_EPOCHES = 50


def run_dataloader(ds: Dataset, batch_size: int, num_epoches: int):
    dl = DataLoader(ds, batch_size, num_workers=min(batch_size, 8), pin_memory=True)
    # warmup
    print("warmup ...")
    for _ in range(4):
        for _ in dl:
            continue
    print("done!")
    with get_torch_profiler() as prof:
        with record_function("run_dataloader"):
            for _ in range(num_epoches):
                for _ in dl:
                    continue
    return prof


def run_model_train(
    ds: Dataset,
    batch_size: int,
    model: nn.Module,
    img_size: Tuple[int, int],
    num_epoches: int = 1,
    do_warmup: bool = True,
):
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scaler = GradScaler()

    dl = DataLoader(
        ds, batch_size=batch_size, num_workers=min(batch_size, 8), pin_memory=True
    )
    criterion = nn.CrossEntropyLoss()
    if do_warmup:
        print("warming up... ")
        run_model_train(ds, batch_size, model, img_size, num_epoches=4, do_warmup=False)
        print("done!")

    with get_torch_profiler() as prof:
        with record_function("run_model_train"):
            for _ in range(num_epoches):
                for images, labels in dl:
                    images = torch.nn.functional.interpolate(images, size=(img_size))
                    images = images[:, :3, :, :].cuda().half()
                    labels = labels.cuda()

                    optimizer.zero_grad()

                    with autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
    return prof


def do_benchmark():
    models = {"resnet18": resnet18, "resnet50": resnet50}
    datasets_prefix = ["uncompressed", "compressed_q:v3"]
    for dataset_prefix in datasets_prefix:
        for img_size in IMAGE_SIZES:
            img_size_str = f"{img_size[0]}-{img_size[1]}"
            ds = FolderDataset(Path(f"{dataset_prefix}_{img_size_str}"))
            for batch_size in BATCH_SIZES:
                benchmark = f"{dataset_prefix}_{img_size_str}_{batch_size}"
                print(f"Running {benchmark}")
                prof = run_dataloader(ds, batch_size, num_epoches=NUM_EPOCHES)
                keys = prof.key_averages()
                my_keys = list(filter(lambda e: e.key in ["run_dataloader"], keys))
                for key in my_keys:
                    logger.info(
                        {
                            "key": "compressed_vs_uncompressed",
                            "benchmark": dataset_prefix,
                            "batch_size": batch_size,
                            "img_size": ",".join([str(e) for e in img_size]),
                            "name": key.key,
                            "time": (key.cpu_time_total + key.cuda_time_total)
                            / 1000
                            / NUM_EPOCHES,
                        }
                    )


do_benchmark()
#  jq 'select(.message.key == "compressed_vs_uncompressed")' logs/my_app_compressed_vs_uncompressed.log.jsonl -c |
#  jq '.message' -c  > temp.jsonl && \
#  python scripts/make_plots.py --input_file temp.jsonl --aggregation_keys benchmark key name batch_size img_size --group_by_name img_size  benchmark --x_axis batch_size --y_axis time   --name compressed_vs_uncompressed --output_dir plots