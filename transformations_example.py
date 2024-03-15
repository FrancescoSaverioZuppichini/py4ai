import torch
import torchvision.transforms as T 

x = torch.zeros((4, 3, 256, 256), dtype=torch.uint8)
# they don't usually change the dtype, expecially if they just move pixels
print(T.RandomCrop((224, 225))(x).dtype)
print(T.Resize((224, 225))(x).dtype)
print(T.RandomRotation(degrees=45)(x).dtype)
print(T.GaussianBlur(kernel_size=3)(x).dtype)
# but normalisation no, you need to pass float
print(T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(x.half()).dtype)

from torch import nn 
from torchvision.models import resnet18

model = nn.Sequential(T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]), resnet18()).half().cuda()
# nn.Sequential is life, nn.Sequential is love
print(model(x.cuda().half()).size())
# if you export the model to onnx, gg you don't need to add weird code to normalise