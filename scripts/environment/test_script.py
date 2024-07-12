import torch
import torch.nn as nn

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version())

# Enable cuDNN benchmarking for potential performance improvements
torch.backends.cudnn.benchmark = True

# Create a simple tensor and move it to GPU
x = torch.randn(10, 10)
if torch.cuda.is_available():
    x = x.cuda()
    print("Tensor on CUDA:", x.device)

# Create a simple model with a convolutional layer
class SimpleConvModel(nn.Module):
    def __init__(self):
        super(SimpleConvModel, self).__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

model = SimpleConvModel()

# Dummy input with correct dimensions [batch_size, channels, height, width]
x = torch.randn(1, 3, 24, 24)

print("Running on CPU")
# Force everything to run on CPU to see if the issue is with CUDA/cuDNN
model.cpu()
x = x.cpu()

try:
    output = model(x)
    print("Output on CPU:", output.shape)
except Exception as e:
    print("Error during model execution on CPU:", e)

# Check if CUDA is available and if so, move the model and data to GPU
if torch.cuda.is_available():
    model = model.cuda()
    x = x.cuda()
    print("Using CUDA")

# Forward pass to compute outputs on GPU
try:
    output = model(x)
    print("Output on GPU:", output.shape)  # Print the shape of the output to confirm it is correct
except Exception as e:
    print("Error during model execution on GPU:", e)
