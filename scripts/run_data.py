import os
import sys
import time
import subprocess

batch_size = 8

def main():
    with open(os.devnull, 'w') as fnull:
        p_server = subprocess.Popen(['python','PipeSwitch/pipeswitch/main.py','PipeSwitch/pipeswitch/model_list.txt'], stdout=fnull, stderr=fnull)
        time.sleep(30)
        p_client = subprocess.Popen(['python','PipeSwitch/client/client_switching.py', 'inception_v3', str(batch_size)], stderr=fnull)
        while True:
            try:
                p_client.wait(1)
                break
            except:
                print (time.time())
                sys.stdout.flush()
            continue
        p_server.kill()

if __name__ == '__main__':
    main()