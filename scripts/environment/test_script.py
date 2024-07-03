import torch
import torch.nn as nn

# Create a simple model with a batch normalization layer
model = nn.Sequential(
    nn.Conv2d(3, 10, kernel_size=3, padding=1),
    nn.BatchNorm2d(10)
)

# Dummy input - this should be similar in shape to what your actual data looks like
# Here, it's a batch of images with 1 batch size, 3 color channels, and 24x24 pixels
x = torch.randn(1, 3, 24, 24)

# Check if CUDA is available and if so, move the model and data to GPU
if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()
    print("Using CUDA")

# Forward pass to compute outputs
try:
    output = model(x)
    print("Output:", output)
except Exception as e:
    print("Error during model execution:", e)

