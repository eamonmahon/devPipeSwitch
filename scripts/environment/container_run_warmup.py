from os import name
from task.resnet152_training import import_data_loader, import_model, import_func
from task.helper import get_data, get_model
import torch

def warm_up_training():
    # Warm up training
    data_loader = import_data_loader()
    model = import_model().cuda()
    func = import_func()
    func(model, data_loader)

def warm_up_inference(model_name):
    # Warm up training
    data = get_data(model_name, 8)
    model, func = get_model(model_name)

    model = model.cuda()
    data_b = data.numpy().tobytes()
    func(model, data_b)

def main():
    warm_up_training()

    # Warmup inference
    # for model_name in ['resnet152', 'inception_v3', 'bert_base']:
    for model_name in ['inception_v3']:
        warm_up_inference(model_name)
    


if __name__ == '__main__':

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    main()