import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("❌ Error: PyTorch cannot see your GPU.")
    print("Fix: You need to install the 'cu121' (CUDA) version of PyTorch.")