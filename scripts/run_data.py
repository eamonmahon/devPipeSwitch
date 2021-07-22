import os
import time
import subprocess

batch_size = 8

def main():
    with open(os.devnull, 'w') as fnull:
        print ('Run 1')
        # p_server = subprocess.Popen(['python','PipeSwitch/kill_restart/kill_restart.py','resnet152'], stdout=fnull, stderr=fnull)
        p_server = subprocess.Popen(['python','PipeSwitch/kill_restart/kill_restart.py','resnet152'])
        print ('Run 2')
        time.sleep(30)
        # p_client = subprocess.Popen(['python','PipeSwitch/client/client_switching.py', 'resnet152', str(batch_size)], stderr=fnull)
        p_client = subprocess.Popen(['python','PipeSwitch/client/client_switching.py', 'resnet152', str(batch_size)])
        print ('Run 3')
        p_client.wait()
        print ('Run 4')
        p_server.kill()
        print ('Run 5')

if __name__ == '__main__':
    main()