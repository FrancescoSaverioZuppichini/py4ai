import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import os
from src.utils import get_logger, get_torch_profiler, schedule
import sys

os.environ["TORCH_COMPILE_DEBUG"] = "1"
batch_sizes = [1, 4, 8, 16, 32, 64]

logger = get_logger()


def run_gpu_first(batch_size):
    inputs = (torch.rand((batch_size, 3, 640, 480)) * 255).to(dtype=torch.uint8)
    with get_torch_profiler() as prof:
        with record_function("to_gpu"):
            inputs = inputs.cuda()
            inputs = inputs.to(torch.float16)
    return prof


def run_gpu_after(batch_size):
    inputs = (torch.rand((batch_size, 3, 640, 480)) * 255).to(dtype=torch.uint8)
    with get_torch_profiler() as prof:
        with record_function("to_gpu"):
            inputs = inputs.to(torch.float16)
            inputs = inputs.cuda()
    return prof


def do_benchmark():
    for _ in range(4):
        run_gpu_first(4)
    for batch_size in batch_sizes:
        prof = run_gpu_first(batch_size)
        keys = prof.key_averages()
        my_keys = list(filter(lambda e: e.key in ["to_gpu"], keys))
        # print(prof.key_averages().table(row_limit=10, top_level_events_only=True))
        for key in my_keys:
            logger.info(
                {
                    "key": "image_to_cuda",
                    "benchmark": "gpu_first",
                    "batch_size": batch_size,
                    "name": key.key,
                    "time": (key.cpu_time_total + key.cuda_time_total) / 1000,
                }
            )
    for batch_size in batch_sizes:
        prof = run_gpu_after(batch_size)
        keys = prof.key_averages()
        my_keys = list(filter(lambda e: e.key in ["to_gpu"], keys))
        # print(prof.key_averages().table(row_limit=10, top_level_events_only=True))
        for key in my_keys:
            logger.info(
                {
                    "key": "image_to_cuda",
                    "benchmark": "gpu_after",
                    "batch_size": batch_size,
                    "name": key.key,
                    "time": (key.cpu_time_total + key.cuda_time_total) / 1000,
                }
            )

    # prof.export_chrome_trace("trace.json")


do_benchmark()
# jq 'select(.message.key == "image_to_cuda")' logs/my_app.log.jsonl -c |
# jq '.message' -c  > temp.jsonl && \
# python scripts/make_plots.py --input_file temp.jsonl --aggregation_keys benchmark key name batch_size --group_by_name benchmark --x_axis batch_size --y_axis time --y_axis_lim 0  --name image_to_cuda --output_dir plots
