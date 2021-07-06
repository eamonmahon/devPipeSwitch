"""
only consider resnet152
"""
import json
import sys
import time
import struct
import statistics

from task.helper import get_data
from core.util import TcpClient, timestamp

def send_request(client, task_name, data):
    timestamp('client', 'before_request_%s' % task_name)

    # Serialize data
    task_name_b = task_name.encode()
    task_name_length = len(task_name_b)
    task_name_length_b = struct.pack('I', task_name_length)

    if data is not None:
        data_b = data.numpy().tobytes()
        length = len(data_b)
    else:
        data_b = None
        length = 0
    length_b = struct.pack('I', length)
    timestamp('client', 'after_inference_serialization')

    # Send Data
    client.send(task_name_length_b)
    client.send(task_name_b)
    client.send(length_b)
    if data_b is not None:
        client.send(data_b)
    timestamp('client', 'after_request_%s' % task_name)

def recv_response(client):
    reply_b = client.recv(4)
    reply = reply_b.decode()
    timestamp('client', 'after_reply')

def close_connection(client):
    model_name_length = 0
    model_name_length_b = struct.pack('I', model_name_length)
    client.send(model_name_length_b)
    timestamp('client', 'close_connection')


def main():
    # unit second
    time_interval = float(sys.argv[1])

    # total time for each each experiment to run
    interval_count = int(sys.argv[2])

    batch_size = 8
    model_name = 'resnet152'
    task_name_inf = '%s_inference' % model_name

    # Load image
    data = get_data(model_name, batch_size)

    latency_list = []
    inf_throughput_list = []
    for _ in range(interval_count + 2):
        each_exp_latency = []
        inf_throughput = 0
        interval_start_time = time.time()
        while True:
            if time.time() - interval_start_time > interval:
                # end current experiment
                inf_throughput_list.append(inf_throughput)
                latency_list.append(each_exp_latency)
                break

            # Connect
            client_inf = TcpClient('localhost', 12345)
            timestamp('client', 'after_inference_connect')
            time_1 = time.time()

            # Send inference request
            send_request(client_inf, task_name_inf, data)

            # Recv inference reply
            recv_response(client_inf)
            time_2 = time.time()
            latency = (time_2 - time_1) * 1000
            each_exp_latency.append(latency)
            inf_throughput += 1

            close_connection(client_inf)
            timestamp('**********', '**********')

    stable_throughput = throughput_list[2:]
    stable_latency_list = sum(latency_list[2:])
    print ('OpenSourceOutputFlag',
        statistics.mean(stable_throughput), 
        statistics.mean(stable_latency_list), 
        min(stable_latency_list), 
        max(stable_latency_list), 
        sep=', '
    )

if __name__ == '__main__':
    main()
