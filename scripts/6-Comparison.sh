#!/bin/bash
# 小波去噪方法的比较

cuda_visible_devices=2

CUDA_VISIBLE_DEVICES=$cuda_visible_devices python ./data/main_comparison.py
