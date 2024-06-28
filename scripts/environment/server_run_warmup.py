import os
import sys

from scripts.common.util import RunDocker

def main():
    print("From server_run_warmup.py...")
    print("Running: RunDocker('pipeswitch:ready_model', dev) as rd:")
    with RunDocker('pipeswitch:ready_model', 'dev') as rd:
        # Start the server: ready_model
        print("issue check")
        rd.run('python PipeSwitch/scripts/environment/container_run_warmup.py')
        print("no issue")
        # Get and return the data point

if __name__ == '__main__':
    print("hello")
    main()