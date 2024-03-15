from tensordict import TensorDict
import torch

data = TensorDict(
    {
        "key 1": torch.ones(3, 4, 5),
        "key 2": torch.zeros(3, 4, 5, dtype=torch.bool),
    },
    batch_size=[3, 4],
)

print(data)
# TensorDict(
#     fields={
#         key 1: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
#         key 2: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.bool, is_shared=False)},
#     batch_size=torch.Size([3, 4]),
#     device=None,
#     is_shared=False)

data.memmap_("tensors/foo")
data.load_memmap("tensors/foo")