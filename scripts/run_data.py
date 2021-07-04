import os
import time
import subprocess

batch_size = 8

def main():
    p_server = subprocess.Popen(['python','PipeSwitch/pipeswitch/main.py','PipeSwitch/pipeswitch/model_list.txt'], stdout=os.devnull, stderr=os.devnull)
    time.sleep(30)
    p_client = subprocess.Popen(['python','PipeSwitch/client/client_switching.py', 'resnet152', str(batch_size)])
    p_client.wait()
    p_server.kill()

if __name__ == '__main__':
    main()