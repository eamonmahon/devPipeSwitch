# Get current work dir
WORK_DIR=$(pwd)/PipeSwitch
echo "error check1"
# Import global variables
source $WORK_DIR/scripts/config/env.sh
echo "error check2"

PYTHONPATH=$PYTHONPATH:$WORK_DIR python $WORK_DIR/scripts/environment/server_run_warmup.py
echo "error check3"