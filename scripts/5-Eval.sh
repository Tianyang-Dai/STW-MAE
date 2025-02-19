#!/bin/bash
# 评估

cuda_visible_devices=2

CUDA_VISIBLE_DEVICES=$cuda_visible_devices python Eval.py \
    --output_dir './eval/eval_PURE'
