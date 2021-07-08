import os
import time
import subprocess

batch_size = 8

def main():
    os.system("nvidia-cuda-mps-control -d")
    with open(os.devnull, 'w') as fnull:
        p_server = subprocess.Popen(['python','PipeSwitch/mps/server_nonstop.py','resnet152'], stdout=fnull, stderr=fnull)
        time.sleep(30)
        p_client = subprocess.Popen(['python','PipeSwitch/client/client_switching.py', 'resnet152', str(batch_size)], stderr=fnull)
        p_client.wait()
        p_server.kill()

if __name__ == '__main__':
    main()