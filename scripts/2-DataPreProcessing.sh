#!/bin/bash
# 数据预处理：STmap和WaveletMap

cuda_visible_devices=2

# STmap
# CHROM-STMap
CUDA_VISIBLE_DEVICES=$cuda_visible_devices python ./data/STMap.py \
    --STMap_channels 'RGB' \
    --STMap_augmentation 'CHROM' \
    --STMap_name 'STMap_RGB_Align_CSI_'

# POS-STMap
CUDA_VISIBLE_DEVICES=$cuda_visible_devices python ./data/STMap.py \
    --STMap_channels 'RGB' \
    --STMap_augmentation 'POS' \
    --STMap_name 'STMap_RGB_Align_CSI_'

# Filtered-STMap
CUDA_VISIBLE_DEVICES=$cuda_visible_devices python ./data/STMap.py \
    --STMap_channels 'RGB' \
    --STMap_augmentation 'Filtered' \
    --STMap_name 'STMap_RGB_Align_CSI_'

# Original-STMap
CUDA_VISIBLE_DEVICES=$cuda_visible_devices python ./data/STMap.py \
    --STMap_channels 'RGB' \
    --STMap_augmentation 'Original' \
    --STMap_name 'STMap_RGB_Align_CSI_'

# WaveletMap
CUDA_VISIBLE_DEVICES=$cuda_visible_devices python ./data/WaveletMap.py \
    --WaveletMap_channels 'RGB' \
    --WaveletMap_name 'WaveletMap_RGB_Align_CSI'
