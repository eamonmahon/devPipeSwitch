import sys
import struct
import threading
import importlib

import numpy
import torch
import torch.multiprocessing as mp

from util.util import TcpServer, TcpAgent, timestamp
from task.common import kMinBlockSize

NUM_WORKERS = 2

def func_pipeline(pipe, summary):
    for shape_list, param_list, buf_list, mod_list in summary:
        real_size_list = [numpy.prod(shape) for shape in shape_list]
        def calc_padded_size(real_size):
            n_bytes = real_size * 4
            padded_n_bytes = (n_bytes + kMinBlockSize - 1) // kMinBlockSize * kMinBlockSize
            padded_size = padded_n_bytes // 4
            return padded_size
        padded_size_list = [calc_padded_size(size) for size in real_size_list]
        batch_cuda = pipe.recv()
        if batch_cuda is not None:
            t_list = batch_cuda.split(padded_size_list)
            index = 0
            for p in param_list:
                t = t_list[index]
                size = real_size_list[index]
                shape = shape_list[index]
                p.data = t[:size].view(shape)
                index += 1
            for mod, key in buf_list:
                t = t_list[index]
                size = real_size_list[index]
                shape = shape_list[index]
                mod._buffers[key] = t[:size].view(shape)
                index += 1
        mod_list[0].lock_for_parameter.release()

def func_terminate(pipe_term, TERM_SIGNAL):
    while True:
        _ = pipe_term.recv()
        if TERM_SIGNAL[0] == 0:
            pipe_term.send('IDLE')
        else:
            TERM_SIGNAL[0] = 2

def worker_compute(model_list, pipe, pipe_term):
    # Warm up CUDA
    torch.randn(1024).cuda()
    torch.cuda.empty_cache()
    cuda_stream_for_computation = torch.cuda.Stream()

    # Create threads for terminate
    TERM_SIGNAL = [0]
    t_term = threading.Thread(target=func_terminate, args=(pipe_term, TERM_SIGNAL))
    t_term.start()

    # Load models
    def worker_load_model(model_name):
        model_module = importlib.import_module('task.' + model_name)
        model, func, summary = model_module.import_task()
        data_loader = model_module.import_data_loader()

        # Add hook for parameter
        def hook_for_parameter(mod, input):
            if not mod.initialized:
                mod.lock_for_parameter.acquire()
                mod.initialized = True
        for _, _, _, mod_list in summary:
            mod_list[0].lock_for_parameter = threading.Lock()
            mod_list[0].lock_for_parameter.acquire()
            mod_list[0].initialized = False
            mod_list[0].register_forward_pre_hook(hook_for_parameter)

        # Add hook for termination
        def hook_for_termination(mod, input, output):
            torch.cuda.synchronize()
            if TERM_SIGNAL[0] == 2:
                raise Exception('terminate')
        def travel_layer(mod):
            if len(list(mod.children())) == 0:
                mod.register_forward_hook(hook_for_termination)
                mod.register_backward_hook(hook_for_termination)
            else:
                for child in mod.children():
                    travel_layer(child)
        if 'training' in model_name:
            travel_layer(model)

        return model, func, data_loader, summary

    models = {}
    for model_name in model_list:
        models[hash(model_name)] = worker_load_model(model_name)

    # Wait for task 
    while True:
        agent, model_name, data_b = pipe.recv()

        model, func, data_loader, summary = models[hash(model_name)]

        t_pipeline = threading.Thread(target=func_pipeline, args=(pipe, summary,))
        t_pipeline.start()
        
        with torch.cuda.stream(cuda_stream_for_computation):
            if 'training' in model_name:
                agent.send(b'FNSH')
                del agent
                timestamp('server', 'reply')
                try:
                    TERM_SIGNAL[0] = 1
                    output = func(model, data_loader)
                except:
                    pipe_term.send('TERM')
                    TERM_SIGNAL[0] = 0
                    timestamp('server', 'complete')
            else:
                output = func(model, data_b)
                timestamp('server', 'complete')
                agent.send(b'FNSH')
                del agent
                timestamp('server', 'reply')

        del model
        del models[hash(model_name)]
        models[hash(model_name)] = worker_load_model(model_name)

def main():
    model_list_file_name = sys.argv[1]
    model_list = []
    with open(model_list_file_name) as f:
        for line in f.readlines():
            model_list.append((line.strip()))

    # Warm up CUDA
    torch.randn(1024).cuda()
    torch.cuda.empty_cache()
    cuda_stream_for_parameter = torch.cuda.Stream()

    def scheduler_load_model(model_name):
        model_module = importlib.import_module('task.' + model_name)
        batch_list = model_module.import_parameters()
        name_list = [batch[1] for batch in batch_list]
        batch_list = [batch[0].detach().pin_memory() if batch[0] is not None else None for batch in batch_list ]
        
        return batch_list, name_list
    models = {}
    for model_name in model_list:
        models[hash(model_name)] = scheduler_load_model(model_name)

    # Worker
    worker_list = []
    for _ in range(NUM_WORKERS):
        p_parent, p_child = mp.Pipe()
        term_parent, term_child = mp.Pipe()
        p = mp.Process(target=worker_compute, args=(model_list, p_child, term_child))
        p.start()
        worker_list.append((p_parent, term_parent))


    active_worker = -1
    server = TcpServer('localhost', 12345)
    while True:
        # Accept request
        conn, _ = server.accept()
        agent = TcpAgent(conn)

        model_name_length_b = agent.recv(4)
        model_name_length = struct.unpack('I', model_name_length_b)[0]
        model_name_b = agent.recv(model_name_length)
        model_name = model_name_b.decode()
        timestamp('tcp', 'get_name')

        data_length_b = agent.recv(4)
        data_length = struct.unpack('I', data_length_b)[0]
        if data_length > 0:
            data_b = agent.recv(data_length)
        else:
            data_b = None
        timestamp('tcp', 'get_data')
        
        # Schedule
        if active_worker > -1:
            _, cur_term = worker_list[active_worker]
            cur_term.send('TERM')
            _ = cur_term.recv()
        active_worker = (active_worker + 1) % NUM_WORKERS
        new_pipe, _ = worker_list[active_worker]
        new_pipe.send((agent, model_name, data_b))

        # Send parameters
        batch_list, name_list = models[hash(model_name)]
        with torch.cuda.stream(cuda_stream_for_parameter):
            for batch, names in zip(batch_list, name_list):
                if batch is not None:
                    batch_cuda = batch.cuda(non_blocking=True)
                else:
                    batch_cuda = None
                new_pipe.send(batch_cuda)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()