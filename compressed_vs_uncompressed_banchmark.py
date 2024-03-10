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


def run_dataloader(ds: Dataset, batch_size: int):
    dl = DataLoader(ds, batch_size, num_workers=8, pin_memory=True)
    for _ in dl:
        continue
    with get_torch_profiler() as prof:
        with record_function("run_dataloader"):
            for _ in dl:
                continue
    return prof


def run_model_train(ds: Dataset, batch_size: int, model: nn.Module):
    model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scaler = GradScaler()

    dl = DataLoader(ds, batch_size=batch_size, num_workers=8, pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    for images, labels in dl:
        images = images[:, :3, :, :].half().cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        with autocast(device_type="cuda", dtype=torch.float16):
            images = interpolate(
                images, size=(224, 224), mode="bilinear", align_corners=False
            )
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    with get_torch_profiler() as prof:
        with record_function("run_model_train"):
            for images, labels in dl:
                images = images[:, :3, :, :].half().cuda()
                labels = labels.cuda()

                optimizer.zero_grad()

                with autocast(device_type="cuda", dtype=torch.float16):
                    images = interpolate(
                        images, size=(224, 224), mode="bilinear", align_corners=False
                    )
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
    return prof


def do_benchmark():
    ds = FolderDataset(Path("data/uncompressed"))
    ds_compressed = FolderDataset(Path("data/compressed_q:v3"))
    models = {"resnet101": resnet101(), "resnet50": resnet50(), "resnet18": resnet18()}
    for batch_size in BATCH_SIZES:
        prof = run_dataloader(ds, batch_size)
        keys = prof.key_averages()
        my_keys = list(filter(lambda e: e.key in ["run_dataloader"], keys))
        for key in my_keys:
            logger.info(
                {
                    "key": "compressed_vs_uncompressed",
                    "benchmark": "uncompressed",
                    "batch_size": batch_size,
                    "name": key.key,
                    "time": (key.cpu_time_total + key.cuda_time_total) / 1000,
                }
            )

        prof = run_dataloader(ds_compressed, batch_size)
        keys = prof.key_averages()
        my_keys = list(filter(lambda e: e.key in ["run_dataloader"], keys))
        for key in my_keys:
            logger.info(
                {
                    "key": "compressed_vs_uncompressed",
                    "benchmark": "compressed_q:v3",
                    "batch_size": batch_size,
                    "name": key.key,
                    "time": (key.cpu_time_total + key.cuda_time_total) / 1000,
                }
            )
    # lazy xD
    for name, model in models.items():
        for batch_size in [16, 32]:
            print(batch_size, name)
            prof = run_model_train(ds, batch_size, model)
            keys = prof.key_averages()
            my_keys = list(filter(lambda e: e.key in ["run_model_train"], keys))
            for key in my_keys:
                logger.info(
                    {
                        "key": "compressed_vs_uncompressed_model_train",
                        "benchmark": f"uncompressed_{name}",
                        "batch_size": batch_size,
                        "name": key.key,
                        "time": (key.cpu_time_total + key.cuda_time_total) / 1000,
                    }
                )

            prof = run_model_train(ds_compressed, batch_size, model)
            keys = prof.key_averages()
            my_keys = list(filter(lambda e: e.key in ["run_model_train"], keys))
            for key in my_keys:
                logger.info(
                    {
                        "key": "compressed_vs_uncompressed_model_train",
                        "benchmark": f"compressed_q:v3_{name}",
                        "batch_size": batch_size,
                        "name": key.key,
                        "time": (key.cpu_time_total + key.cuda_time_total) / 1000,
                    }
                )


do_benchmark()
# jq 'select(.message.key == "compressed_vs_uncompressed")' logs/my_app.log.jsonl -c |
# jq '.message' -c  > temp.jsonl && \
# python scripts/make_plots.py --input_file temp.jsonl --aggregation_keys benchmark key name batch_size --group_by_name benchmark --x_axis batch_size --y_axis time --y_axis_lim 0  --name compressed_vs_uncompressed --output_dir plots

# jq 'select(.message.key == "compressed_vs_uncompressed_model_train")' logs/my_app.log.jsonl -c |
# jq '.message' -c  > temp.jsonl && \
# python scripts/make_plots.py --input_file temp.jsonl --aggregation_keys benchmark key name batch_size --group_by_name benchmark --x_axis batch_size --y_axis time --y_axis_lim 0  --name compressed_vs_uncompressed --output_dir plots
