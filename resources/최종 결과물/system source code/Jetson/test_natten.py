# Test natten on CPU and GPU/CUDA

# import natten
# print("natten version:", natten.__version__) # 0.17.3
# print(dir(natten))

import torch
from natten import NeighborhoodAttention1D, NeighborhoodAttention2D, NeighborhoodAttention3D

def test_natten_on_device(device):
    try:
        print(f"Testing on {device}...")

        layer = NeighborhoodAttention1D(dim=64, kernel_size=7, num_heads=4).to(device)
        x = torch.randn(1, 16, 64, device=device)
        output = layer(x)
        
        print(f"Success on {device}!")
        return output
    except Exception as e:
        print(f"Error on {device}: {e}")

# Test on CPU
cpu_output = test_natten_on_device("cpu")

# Test on GPU if available
if torch.cuda.is_available():
    gpu_output = test_natten_on_device("cuda")
else:
    print("GPU is not available.")
