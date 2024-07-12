import os
import torch
import torch.nn as nn

# Set environment variables for cuDNN debugging
os.environ['CUDNN_LOGINFO_DBG'] = '1'
os.environ['CUDNN_LOGDEST_DBG'] = 'stdout'

# Function to print system and tensor information
def print_system_info():
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Version:", torch.version.cuda)
    print("cuDNN Version:", torch.backends.cudnn.version())

# Function to test a simple fully connected layer
def test_fully_connected():
    print("\nTesting Fully Connected Layer...")
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 5)  # A single fully connected layer

        def forward(self, x):
            return self.fc(x)

    model = SimpleModel()

    # Dummy input for the fully connected layer
    x = torch.randn(10, 10)  # Batch size of 10, input features of 10

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

# Function to test a simple convolutional layer
def test_convolutional():
    print("\nTesting Convolutional Layer...")
    
    class SimpleConvModel(nn.Module):
        def __init__(self):
            super(SimpleConvModel, self).__init__()
            # Experiment with different kernel sizes, padding, and stride
            self.conv = nn.Conv2d(3, 10, kernel_size=1, padding=0, stride=1)

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

    # Enable cuDNN benchmarking
    torch.backends.cudnn.benchmark = True

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

if __name__ == "__main__":
    print_system_info()
    test_fully_connected()
    test_convolutional()
