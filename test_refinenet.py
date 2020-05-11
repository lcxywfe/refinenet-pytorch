from models.resnet import rf101
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

model = rf101(30, pretrained=False)

dtype = torch.float
device = torch.device("cpu")
x = torch.randn(1, 3, 256, 256, device=device, dtype=dtype)

writer = SummaryWriter('logs/test')
writer.add_graph(model, x)
writer.close()

print(model(x).shape)
