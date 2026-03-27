export PYTHONPATH="$PWD:$PYTHONPATH"
export CFG_FILE="configs/test/dac_dinov3l+dpt_$1_test_$2.json"

python scripts/test_dac.py --model-file ./checkpoints/unidac.pt --model-name UniDAC --config-file $CFG_FILE --base-path datasets