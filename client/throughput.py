"""
only consider resnet152
"""
import json
import sys
import time
import struct
import statistics

from task.helper import get_data
from util.util import TcpClient, timestamp

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
    time_interval_ms = time_interval * 1000

    # total time for each each experiment to run
    interval_count = int(sys.argv[2])

    batch_size = 8
    model_name = 'resnet152'
    task_name_inf = '%s_inference' % model_name
    task_name_train = '%s_training' % model_name

    # Load image
    data = get_data(model_name, batch_size)

    inf_latency_list = []
    inf_throughput_list = []
    cur_task = task_name_train
    for _ in range(interval_count + 2):
        each_exp_latency = []
        inf_throughput = 0
        interval_start_time = time.time()

        last_request = None
        client_train = None
        while True:
            if time.time() - interval_start_time > time_interval:
                # end current experiment
                if last_request == task_name_inf:
                    inf_throughput_list.append(inf_throughput)
                    inf_latency_list.append(each_exp_latency)

                # switch task
                cur_task = task_name_inf if cur_task != task_name_inf else task_name_train
                interval_start = time.time()

                break

            if cur_task == task_name_train:
                # Send training request
                client_train = TcpClient('localhost', 12345)
                send_request(client_train, task_name_train, None)
                time.sleep(time_interval)

                last_request = task_name_train
                print('end the training request')
            else:
                # inference request
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
                # latency_list.append(latency)
                each_exp_latency.append(latency)
                if latency < time_interval_ms:
                    inf_throughput += 1

                # time.sleep(0.1)
                if last_request == task_name_train:
                    recv_response(client_train)
                    close_connection(client_train)
                    client_train = None

                close_connection(client_inf)

                # record last request
                last_request = task_name_inf
                # time.sleep(0.1)
                timestamp('**********', '**********')

    stable_throughput = inf_throughput_list[2:]
    stable_latency_list = sum(inf_latency_list[2:], [])
    print ('OpenSourceOutputFlag',
        statistics.mean(stable_throughput) / time_interval,
        statistics.mean(stable_latency_list), 
        min(stable_latency_list), 
        max(stable_latency_list), 
        sep=', '
    )

if __name__ == '__main__':
    main()
