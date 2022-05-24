import torch

print(torch.__version__)
print(f"cuda, {torch.cuda.is_available()}")
print(f"compute_{''.join(map(str,(torch.cuda.get_device_capability())))}")
device_num:int = torch.cuda.device_count()
print(f"find gpu devices, {device_num}")
for idx in range(device_num):
    print(f"cuda:{idx}, {torch.cuda.get_device_name(idx)}")

print("end")

