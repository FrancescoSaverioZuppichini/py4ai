import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import schedule

model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224).cuda()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("model_inference"):
        model(inputs)

print(
    prof.key_averages(group_by_input_shape=True).table(
        sort_by="cpu_time_total", row_limit=10
    )
)
prof.export_chrome_trace("trace.json")

# # you can also create a schedule
# my_schedule = schedule(
#     skip_first=10,
#     wait=5,
#     warmup=1,
#     active=3,
#     repeat=2)

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=my_schedule) as prof:
#     with record_function("model_inference"):
#         model(inputs)

model = models.resnet18().cuda()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("inputs_to_cuda"):
        inputs = inputs.cuda()
    with record_function("model_inference"):
        model(inputs)

print(
    prof.key_averages(group_by_input_shape=True).table(
        sort_by="cpu_time_total", row_limit=10
    )
)
prof.export_chrome_trace("trace2.json")
