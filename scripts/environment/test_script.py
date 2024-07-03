import torch
import torch.nn as nn

print(torch.version.cuda)
print(torch.backends.cudnn.version())

# Create a simple model with only a convolutional layer
model = nn.Sequential(
    nn.Conv2d(3, 10, kernel_size=3, padding=1)
)

# Dummy input
x = torch.randn(1, 3, 24, 24)

print("running on cpu")
# Force everything to run on CPU to see if the issue is with CUDA/cuDNN
model.cpu()
x = x.cpu()

try:
    output = model(x)
    print("Output:", output.shape)
except Exception as e:
    print("Error during model execution on CPU:", e)

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
    print("Error during model execution on GPU:", e)
