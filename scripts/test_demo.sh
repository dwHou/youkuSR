#!/bin/bash
demo() {
    SAVE=demo-4345
    PRETRAIN='../model/EDVR-F10RG10frame9/youku_800samples.pt'
    export MODEL_TEMPLATE=edvr-c2f
    export CUDA_VISIBLE_DEVICES=0
    params+=" --n_GPUs 1 "
    params+=" --model edvr-rg "
    params+=" --n_frames 9 --front_RBs 10 --n_cagroups 0 --n_resgroups 10 --n_resblocks 10 "
    params+=" --save $SAVE "
    params+=" --pre_train $PRETRAIN "
    params+=" --test_only --save_results "
    params+=" --data_test VideoDemo "
    params+=" --dir_demo /home/yeyy/datasets/demo_video/demo "
    params+=" --not_hr --n_test_frame 0 "
    params+=" --chop "
    params+=" --test_patch_size 512 "
    params+=" --reset "
    python eval.py $params
}

if [ $1 == "demo" ]; then
    demo
fi
