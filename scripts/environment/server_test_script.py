import os
import sys

from scripts.common.util import RunDocker

def main():
    with RunDocker('pipeswitch:ready_model', 'dev') as rd:
        rd.run('python PipeSwitch/scripts/environment/test_script.py')

if __name__ == '__main__':
    main()
