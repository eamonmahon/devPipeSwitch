import os
import time
import subprocess

batch_size = 8

def main():
    with open(os.devnull, 'w') as fnull:
        p_server = subprocess.Popen(['python','PipeSwitch/memory_ipc/memory_ipc.py','PipeSwitch/memory_ipc/model_list.txt'], stdout=fnull, stderr=fnull)
        time.sleep(30)
        p_client = subprocess.Popen(['python','PipeSwitch/client/client_switching.py', 'bert_base', str(batch_size)], stderr=fnull)
        p_client.wait()
        p_server.kill()

if __name__ == '__main__':
    main()