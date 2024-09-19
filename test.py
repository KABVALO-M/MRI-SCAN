import torch
print(torch.__version__)  # Should show the PyTorch version
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.cuda.get_device_name(0))  # Should return the name of your GPU (e.g., MX450)
