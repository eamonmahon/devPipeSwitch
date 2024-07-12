import torch
import torch.nn as nn

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version())

# Create a simple tensor and move it to GPU
x = torch.randn(10, 10)
if torch.cuda.is_available():
    x = x.cuda()
    print("Tensor on CUDA:", x.device)

# Create a simpler model with only a fully connected layer
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)  # A single fully connected layer

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

# Dummy input for the fully connected layer
x = torch.randn(10, 10)  # Batch size of 10, input features of 10

print("running on cpu")
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
