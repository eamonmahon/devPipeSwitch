"""
only consider resnet152
"""
import json
import sys
import time
import struct
import statistics

from experiments.helper import get_data
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
    time_out = float(sys.argv[2])
    logfile = sys.argv[3]

    # model_name = sys.argv[1]
    batch_size = 8
    repeat_n = 5

    model_name = 'resnet152'
    task_name_inf = '%s_inference' % model_name
    task_name_train = '%s_training' % model_name

    # Load image
    data = get_data(model_name, batch_size)

    # tasks = [task_name_train, task_name_inf]
    latency_list = []
    inf_throughput_list = []
    for _ in range(repeat_n):
        exp_start_t = time.time()

        each_exp_latency = []
        cur_task = task_name_train
        last_request = None
        client_train = None
        inf_throughput = 0
        interval_start = time.time()
        while True:
            if time.time() - exp_start_t > time_out:
                # end current experiment
                inf_throughput_list.append(inf_throughput)
                latency_list.append(each_exp_latency)
                break

            if time.time() - interval_start > time_interval:
                # switch task
                cur_task = task_name_inf if cur_task != task_name_inf else task_name_train
                interval_start = time.time()
            else:
                if cur_task == task_name_train:

                    # Send training request
                    client_train = TcpClient('localhost', 12345)
                    send_request(client_train, task_name_train, None)
                    time.sleep(time_interval)

                    last_request = task_name_train
                    print('end the training request')
                else:
                    # inference request

                    print('start sending an inference request')
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

    print()
    print()
    print()
    # stable_latency_list = latency_list[10:]
    # print (stable_latency_list)
    # print ('Latency: %f ms (stdev: %f)' % (statistics.mean(stable_latency_list), 
    #                                        statistics.stdev(stable_latency_list)))
    with open(logfile, 'w') as ofile:
        res = {
            'time_interval': time_interval,
            'time_out': time_out,
            'throughput': inf_throughput_list,
            'latency': latency_list
            }
        json.dump(res, ofile, indent=2)
    print(latency_list)

    print('throughput list', inf_throughput_list)
    print('average throughput', statistics.mean(inf_throughput_list) )

if __name__ == '__main__':
    main()
