import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders
from tqdm import tqdm
import os
from models.Models_xwj import Siam_NestedUNet_Conc, SNUNet_ECAM_XWJ
import torch
import torch.nn as nn
from thop import profile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_tensor1 = torch.randn(1, 3, 256, 256).to(device)
input_tensor2 = torch.randn(1, 3, 256, 256).to(device)

# model = SNUNet_ECAM().to(device)
model = SNUNet_ECAM_XWJ().to(device)
print("#"*50)
# print(model)
print("#"*50)
# model = Siam_NestedUNet_Conc().to(device)

flops, params = profile(model, inputs=(input_tensor1, input_tensor2))
print(f"FLOPs: {flops}, Params: {params}")

flops_giga = flops / (10 ** 9)
params_mega = params / (10 ** 6)

print(f"FLOPs: {flops_giga} G, Params: {params_mega} M")

