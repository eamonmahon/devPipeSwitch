import os
import time
import subprocess

batch_size = 8

def main():
    os.system("nvidia-cuda-mps-control -d")
    with open(os.devnull, 'w') as fnull:
        p_server = subprocess.Popen(['python','PipeSwitch/mps/server_nonstop.py','resnet152'], stdout=fnull, stderr=fnull)
        time.sleep(30)

        scheduling_cycle = 1
        interval_count = 10
        p_client = subprocess.Popen(['python', 'PipeSwitch/client/throughput.py', str(scheduling_cycle), str(interval_count)], stderr=fnull)

        p_client.wait()
        p_server.kill()

if __name__ == '__main__':
    main()