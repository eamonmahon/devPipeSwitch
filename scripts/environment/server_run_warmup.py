import os
import sys

from scripts.common.util import RunDocker

def main():
    print("Running: RunDocker('pipeswitch:ready_model', dev) as rd:")
    with RunDocker('pipeswitch:ready_model', 'dev') as rd:
        # Start the server: ready_model
        rd.run('python PipeSwitch/scripts/environment/container_run_warmup.py')
        
        # Get and return the data point

if __name__ == '__main__':
    main()