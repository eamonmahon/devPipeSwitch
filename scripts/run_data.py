import os
import time
import subprocess

batch_size = 8

def main():
    with open(os.devnull, 'w') as fnull:
        p_server = subprocess.Popen(['python','PipeSwitch/ready_model/ready_model.py','bert_base'], stdout=fnull, stderr=fnull)
        time.sleep(30)
        p_client = subprocess.Popen(['python','PipeSwitch/client/client_inference.py', 'bert_base', str(batch_size)], stderr=fnull)
        p_client.wait()
        p_server.kill()

if __name__ == '__main__':
    main()