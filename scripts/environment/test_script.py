import torch
import torch.nn as nn

# Create a simple model with only a convolutional layer
model = nn.Sequential(
    nn.Conv2d(3, 10, kernel_size=3, padding=1)
)

# Dummy input
x = torch.randn(1, 3, 24, 24)

# Check if CUDA is available and if so, move the model and data to GPU
if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()
    print("Using CUDA")

# Forward pass to compute outputs
try:
    output = model(x)
    print("Output:", output.shape)  # Print the shape of the output to confirm it is correct
except Exception as e:
    print("Error during model execution:", e)
