import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import os
from src.utils import get_logger, get_torch_profiler, schedule
from torch.utils.data import DataLoader, Dataset
from src.dataset import DummyDataset
from pathlib import Path
from torch import nn
from torch import optim, autocast
from torch.cuda.amp import GradScaler
from torchvision.models import resnet101, resnet50, resnet18
from torch.nn.functional import interpolate
from time import perf_counter

os.environ["TORCH_COMPILE_DEBUG"] = "1"

logger = get_logger()
IMAGE_SIZES = [(384, 384), (640, 480), (1280, 960)]
BATCH_SIZES = [8, 16, 32, 64]
NUM_EPOCHES = 10


def run_model_train(
    ds: Dataset,
    batch_size: int,
    model: nn.Module,
    cuda_first: bool = True,
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
        run_model_train(
            ds, batch_size, model, cuda_first, num_epoches=4, do_warmup=False
        )
        print("done!")

    with get_torch_profiler() as prof:
        with record_function("run_model_train"):
            for _ in range(num_epoches):
                for images, labels in dl:
                    if cuda_first:
                        images = images.cuda().half()
                    else:
                        images = images.half().cuda()
                    labels = labels.cuda()

                    optimizer.zero_grad()

                    with autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                end = perf_counter()
    return prof


def do_benchmark():
    models = {"resnet18": resnet18, "resnet50": resnet50}
    for name, model_func in models.items():
        for img_size in IMAGE_SIZES:
            for batch_size in BATCH_SIZES:
                ds = DummyDataset(num_images=256, img_size=img_size)
                model = model_func()
                benchmark = f"{name}_{batch_size}_{img_size}_cuda_first=True"
                print(f"Running {benchmark}")
                prof = run_model_train(
                    ds, batch_size, model, cuda_first=True, num_epoches=NUM_EPOCHES
                )
                keys = prof.key_averages()
                my_keys = list(filter(lambda e: e.key in ["run_model_train"], keys))
                for key in my_keys:
                    logger.info(
                        {
                            "key": "image_to_cuda_model_train",
                            "benchmark": benchmark,
                            "batch_size": batch_size,
                            "cuda_first": True,
                            "model": name,
                            "name": key.key,
                            "img_size": ",".join([str(e) for e in img_size]),
                            "time": (key.cpu_time_total + key.cuda_time_total)
                            / 1000
                            / NUM_EPOCHES,
                            "img/ms": (key.cpu_time_total + key.cuda_time_total)
                            / 1000
                            / NUM_EPOCHES
                            / ds.num_images,
                        }
                    )
                benchmark = f"{name}_{batch_size}_{img_size}_cuda_first=False"
                prof = run_model_train(
                    ds, batch_size, model, cuda_first=False, num_epoches=NUM_EPOCHES
                )
                keys = prof.key_averages()
                my_keys = list(filter(lambda e: e.key in ["run_model_train"], keys))
                for key in my_keys:
                    logger.info(
                        {
                            "key": "image_to_cuda_model_train",
                            "benchmark": benchmark,
                            "batch_size": batch_size,
                            "cuda_first": False,
                            "model": name,
                            "name": key.key,
                            "img_size": ",".join([str(e) for e in img_size]),
                            "time": (key.cpu_time_total + key.cuda_time_total)
                            / 1000
                            / NUM_EPOCHES,
                            "img/ms": (key.cpu_time_total + key.cuda_time_total)
                            / 1000
                            / NUM_EPOCHES
                            / ds.num_images,
                        }
                    )


do_benchmark()
# jq 'select(.message.key == "image_to_cuda_model_train")' logs/my_app.log.jsonl -c |
# jq '.message' -c  > temp.jsonl && \
# python scripts/make_plots.py --input_file temp.jsonl --aggregation_keys model img_size benchmark key name batch_size --group_by_name benchmark --x_axis batch_size --y_axis time --y_axis_lim 0  --name image_to_cuda_model_train --output_dir plots
