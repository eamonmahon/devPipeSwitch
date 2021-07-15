import sys
import queue
import struct
import threading
import importlib
import traceback

import numpy
import torch
import torch.multiprocessing as mp

from util.util import TcpServer, TcpAgent, timestamp
from task.common import kMinBlockSize

NUM_WORKERS = 2

def func_get_request(qout):
    # Listen connections
    server = TcpServer('localhost', 12345)

    while True:
        # Get connection
        conn, _ = server.accept()
        agent = TcpAgent(conn)

        model_name_length_b = agent.recv(4)
        model_name_length = struct.unpack('I', model_name_length_b)[0]
        if model_name_length == 0:
            break
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
        qout.put((agent, model_name, data_b))

def func_schedule(qin, worker_list):
    active_worker = -1
    while True:
        agent, model_name, data_b = qin.get()
        if active_worker >= 0:
            cur_pipe, cur_term_pipe = worker_list[active_worker]
            cur_term_pipe.send('TERM')
            cur_term_pipe.recv()
        active_worker = (active_worker + 1) % NUM_WORKERS
        new_pipe, _ = worker_list[active_worker]
        new_pipe.send((agent, model_name, data_b))
        

def func_terminate(pipe_term, TERM_SIGNAL):
    while True:
        _ = pipe_term.recv()
        if TERM_SIGNAL[0] == 0:
            pipe_term.send('IDLE')
        else:
            TERM_SIGNAL[0] = 2

def func_pipeline(cuda_stream, info_list):
    with torch.cuda.stream(cuda_stream):
        for batch, layer_list, padded_size_list, real_size_list, shape_list in info_list:
            index = 0
            if batch is not None:
                batch = batch.cuda(non_blocking=True)
                t_list = batch.split(padded_size_list)
                for layer in layer_list:
                    for p in layer.parameters():
                        t = t_list[index]
                        real_size = real_size_list[index]
                        shape = shape_list[index]
                        p.data = t[:real_size].view(shape)
                        index += 1
                for layer in layer_list:
                    for key, buf in layer._buffers.items():
                        if buf is not None and buf.dtype is torch.float32:
                            t = t_list[index]
                            real_size = real_size_list[index]
                            shape = shape_list[index]
                            layer._buffers[key] = t[:real_size].view(shape)
                            index += 1
                        else:
                            layer._buffers[key] = None
            layer_list[0].event_for_trans = torch.cuda.Event()
            layer_list[0].event_for_trans.record()
            layer_list[0].lock_for_param.release()

def worker_compute(model_list, pipe, pipe_term):
    # Warm Up CUDA
    torch.randn(1024).cuda()
    torch.cuda.empty_cache()

    cuda_stream_for_param = torch.cuda.Stream()
    cuda_stream_for_comp = torch.cuda.Stream()

    TERM_SIGNAL = [0]
    t_term = threading.Thread(target=func_terminate, args=(pipe_term, TERM_SIGNAL))
    t_term.start()

    def load_model(model_name):
        model_module = importlib.import_module('task.' + model_name)
        model, func, summary_list = model_module.import_task()
        data_loader = model_module.import_data_loader()
        batch_list = model_module.import_parameters()

        # Prepare for pipelines
        info_list = []
        for batch, summary in zip(batch_list, summary_list):
            if batch[0] is not None:
                batch = batch[0].detach().pin_memory()
            else:
                batch = None
            shape_list, _, _, layer_list = summary
            real_size_list = [numpy.prod(shape) for shape in shape_list]
            def calc_padded_size(real_size):
                n_bytes = real_size * 4
                padded_n_bytes = (n_bytes + kMinBlockSize - 1) // kMinBlockSize * kMinBlockSize
                padded_size = padded_n_bytes // 4
                return padded_size
            padded_size_list = [calc_padded_size(size) for size in real_size_list]
            info_list.append((batch, layer_list, padded_size_list, real_size_list, shape_list))

        # Add hook to sync pipeline
        def hook_for_parameter(mod, input):
            if not mod.initialized:
                mod.lock_for_param.acquire()
                mod.event_for_trans.synchronize()
                mod.initialized = True
        for _, layer_list, _, _, _ in info_list:
            layer_list[0].lock_for_param = threading.Lock()
            layer_list[0].lock_for_param.acquire()
            layer_list[0].register_forward_pre_hook(hook_for_parameter)
            layer_list[0].initialized = False

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

        return (model, func, data_loader, info_list)
    
    models = {}
    for model_name in model_list:
        models[hash(model_name)] = load_model(model_name)

    while True:
        agent, model_name, data_b = pipe.recv()
        model, func, data_loader, info_list = models[hash(model_name)]

        # Create a thread to transfer model
        t_param = threading.Thread(target=func_pipeline, args=(cuda_stream_for_param, info_list))
        t_param.start()

        with torch.cuda.stream(cuda_stream_for_comp):
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
                    timestamp('server', 'terminate')
            else:
                output = func(model, data_b)
                print ('Inference output', output)
                timestamp('server', 'complete')
                agent.send(b'FNSH')
                del agent
                timestamp('server', 'reply')

        model.to('cpu')
        del models[hash(model_name)]
        models[hash(model_name)] = load_model(model_name)
        torch.cuda.empty_cache()
        print (torch.cuda.memory_allocated(), torch.cuda.memory_cached())

    

def main():
    # Get model name
    model_list_file_name = sys.argv[1]
    model_list = []
    with open(model_list_file_name) as f:
        for line in f.readlines():
            model_list.append(line.strip())

    # Create workers
    worker_list = []
    for _ in range(NUM_WORKERS):
        p_parent, p_child = mp.Pipe()
        term_parent, term_child = mp.Pipe()
        worker_list.append((p_parent, term_parent))
        p_compute = mp.Process(target=worker_compute, args=(model_list, p_child, term_child))
        p_compute.start()

    # Create TCP server and scheduler
    q_to_schedule = queue.Queue()
    t_get = threading.Thread(target=func_get_request, args=(q_to_schedule,))
    t_get.start()
    t_schedule = threading.Thread(target=func_schedule, args=(q_to_schedule, worker_list))
    t_schedule.start()

    # Accept connection
    t_get.join()
    t_schedule.join()
    

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()