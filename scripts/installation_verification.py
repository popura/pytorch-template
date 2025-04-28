import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is CUDA available?: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA compute capability: {''.join(map(str,(torch.cuda.get_device_capability())))}")
device_num:int = torch.cuda.device_count()
print(f"# of GPU devices: {device_num}")
for idx in range(device_num):
    print(f"cuda:{idx}, {torch.cuda.get_device_name(idx)}")

print("end")
