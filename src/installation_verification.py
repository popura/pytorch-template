import torch

print(torch.__version__)
print(f"Is CUDA available?: {torch.cuda.is_available()}")
print(f"CUDA compute capability: {''.join(map(str,(torch.cuda.get_device_capability())))}")
device_num:int = torch.cuda.device_count()
print(f"# of GPU devices: {device_num}")
for idx in range(device_num):
    print(f"cuda:{idx}, {torch.cuda.get_device_name(idx)}")

print("end")
<<<<<<< HEAD
=======

>>>>>>> 5fa0a532c54cbae67f18ebb19d303e9c862bf757
