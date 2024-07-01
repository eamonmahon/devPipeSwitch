import os
import sys

from scripts.common.util import RunDocker

def main():
    with RunDocker('pipeswitch:ready_model', 'dev') as rd:
        # Set the environment variable and run the script in the same command
        rd.run('CUDA_LAUNCH_BLOCKING=1 python PipeSwitch/scripts/test_script.py')

if __name__ == '__main__':
    main()
