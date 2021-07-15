import sys
import queue
import struct
import threading
import importlib

import torch
import torch.multiprocessing as mp

from experiments.helper import get_model
from core.util import TcpServer, TcpAgent, timestamp

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

def func_schedule(qin, pipe, pipe_term):
    while True:
        agent, model_name, data_b = qin.get()
        pipe_term.send('TERM')
        _ = pipe_term.recv()
        pipe.send((agent, model_name, data_b))

def func_terminate(pipe_term, TERM_SIGNAL):
    while True:
        _ = pipe_term.recv()
        if TERM_SIGNAL[0] == 0:
            pipe_term.send('IDLE')
        else:
            TERM_SIGNAL[0] = 2

def worker_compute(model_list, pipe, pipe_term):
    # Warm Up CUDA
    torch.randn(1024).cuda()

    TERM_SIGNAL = [0]
    t_term = threading.Thread(target=func_terminate, args=(pipe_term, TERM_SIGNAL))
    t_term.start()

    models = {}
    for model_name in model_list:
        model_module = importlib.import_module('task.' + model_name)
        model, func, _ = model_module.import_task()
        data_loader = model_module.import_data_loader()

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

        models[hash(model_name)] = (model, func, data_loader)

    while True:
        agent, model_name, data_b = pipe.recv()
        model, func, data_loader = models[hash(model_name)]
        model.to('cuda')

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

        model.to('cpu')

def main():
    # Get model name
    model_list_file_name = sys.argv[1]

    model_list = []
    with open(model_list_file_name) as f:
        for line in f.readlines():
            model_list.append(line.strip())

    # Create threads and worker process
    q_to_schedule = queue.Queue()
    p_parent, p_child = mp.Pipe()
    term_parent, term_child = mp.Pipe()
    t_get = threading.Thread(target=func_get_request, args=(q_to_schedule, ))
    t_get.start()
    t_schedule = threading.Thread(target=func_schedule, args=(q_to_schedule, p_parent, term_parent))
    t_schedule.start()
    p_compute = mp.Process(target=worker_compute, args=(model_list, p_child, term_child))
    p_compute.start()

    # Accept connection
    t_get.join()
    t_schedule.join()
    p_compute.join()
    

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()