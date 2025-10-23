import torch
import torch.nn as nn
from thop import profile, clever_format
from torchinfo import summary as torchinfo_summary
from models.HibriMambaBlock import *

if __name__ == '__main__':

    model = HibriMambaBlock(embed_dim=32)

    input_size = (3, 128, 128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    torchinfo_summary(model, input_size=(1, *input_size), device=device.type)

    input_tensor = torch.randn(1, *input_size).to(device)  
    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print(f"Params: {params}")
    print(f"FLOPs: {flops}")