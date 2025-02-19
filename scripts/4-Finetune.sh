#!/bin/bash
# 微调

cuda_visible_devices=2

wandb disabled  # 关闭wandb

CUDA_VISIBLE_DEVICES=$cuda_visible_devices python main_finetune.py \
	--log './finetune/finetune_PURE' \
	--log_dir './finetune/finetune_PURE' \
	--output_dir './finetune/finetune_PURE' \
	--finetune './pretrain/pretrain_PURE/checkpoint-399.pth' \
	--Map_name1 'STMap_RGB_Align_CSI_CHROM.png' \
    --Map_name2 'WaveletMap_RGB_Align_CSI.png' \
	--loss_type 'rppg' \
	--dataname 'PURE' \
	--in_chans 6 \
	--nb_classes 224 \
	--fold_num 5 \
	--reData 0
	