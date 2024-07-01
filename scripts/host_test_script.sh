WORK_DIR=$(pwd)

source $WORK_DIR/scripts/config/env.sh

PYTHONPATH=$PYTHONPATH:$WORK_DIR python scripts/host_test_script.py $WORK_DIR/scripts/config/servers.txt
