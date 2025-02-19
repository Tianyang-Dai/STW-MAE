#!/bin/bash
# 预训练

cuda_visible_devices=2

CUDA_VISIBLE_DEVICES=$cuda_visible_devices python main_pretrain.py \
    --log './pretrain/pretrain_PURE' \
    --log_dir './pretrain/pretrain_PURE' \
    --output_dir './pretrain/pretrain_PURE' \
    --reTrain 0 \
    --reData 1 \
    --dataname 'PURE' \
    --Map_name1 'STMap_RGB_Align_CSI_CHROM.png' \
    --Map_name2 'WaveletMap_RGB_Align_CSI.png' \
    --loss_type 'CEP' \
    --in_chans 6 \
    --mask_ratio 0.8 \
    --decoder_embed_dim 128 \
    --decoder_depth 8 \
    --batch_size 256
