import sys
import queue
import struct
import threading
import importlib

import torch.multiprocessing as mp

# from core.checkpoint_server import CheckpointProc
from util.util import TcpServer, TcpAgent, timestamp
from sys import argv

def func_get_request(qout):
    # Start checkpoint serser
    # ckpt = CheckpointProc()
    # ckpt.start()

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

def func_schedule(qin):
    active_worker = None
    active_task = None
    active_pipe = None
    while True:
        agent, model_name, data_b = qin.get()
        timestamp('schedule get model name', model_name)
        if model_name == active_task:
            active_pipe.send((agent, model_name, data_b))
        else:
            if active_worker is not None:
                timestamp('Killing active worker', '')
                active_worker.terminate()
                active_worker.join()
            parent_p, child_p = mp.Pipe()
            active_pipe = parent_p
            active_task = model_name
            active_worker = mp.Process(target=worker_compute, args=(agent, model_name, data_b, child_p))
            active_worker.start()
            timestamp('schedule', 'new worker started for {}'.format(model_name))

def worker_compute(agent, model_name, data_b, pipe):
    # Load model
    model_module = importlib.import_module('task.' + model_name)
    model, func, _ = model_module.import_task()
    data_loader = model_module.import_data_loader()

    timestamp('worker comp proc', 'model loaded to cpu')

    # Model to GPU
    model = model.to('cuda')
    timestamp('****workr', 'loaded to cuda')

    # Compute
    if 'training' in model_name:
        agent.send(b'FNSH')
        del agent
        timestamp('server', 'reply')

        output = func(model, data_loader)
        timestamp('server', 'complete')

    else:
        output = func(model, data_b)
        timestamp('server', 'complete')

        agent.send(b'FNSH')
        del agent
        timestamp('server', 'complete inf ')

    while True:
        # for continous inference task
        agent, model_name, data_b = pipe.recv()
        output = func(model, data_b)
        agent.send(b'FNSH')
        del agent
        timestamp('server', 'complete inf and reply')

def main():
    # Create threads and worker process
    q_to_schedule = queue.Queue()
    t_get = threading.Thread(target=func_get_request, args=(q_to_schedule,))
    t_get.start()
    t_schedule = threading.Thread(target=func_schedule, args=(q_to_schedule,))
    t_schedule.start()

    # Accept connection
    t_get.join()
    t_schedule.join()
    

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
