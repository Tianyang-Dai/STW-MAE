import cv2
import os
import numpy as np
import shutil
import pandas as pd
import scipy.io as scio
from scipy import interpolate
import scipy.io as io
import json
import argparse
from pathlib import Path
from tqdm import trange
from datetime import datetime


def get_args_parser():
    parser = argparse.ArgumentParser('STW-MAE eval for image classification', add_help=False)
    parser.add_argument('--output_dir', default='./supervise_VIT_VIPL_LossCrossEntropy',
                        help='path where to save, empty for no saving')
    
    return parser


def read_ground_truth(json_path):
    with open(json_path, 'r') as infile:
        gt_data = json.load(infile)
  
    video_t = []
    for sample in gt_data['/Image']:
        video_t.append(sample['Timestamp'])
  
    wave_t = []
    bvp = []
    gt = []
    for sample in gt_data['/FullPackage']:
        wave_t.append(sample['Timestamp'])
        bvp.append(sample['Value']['waveform'])
        gt.append(sample['Value']['pulseRate'])
  
    video_t = np.array(video_t)*1e-9
    wave_t = np.array(wave_t)*1e-9
    bvp = np.array(bvp)
    bvp = np.interp(video_t, wave_t, bvp)
    gt = np.array(gt)
    gt = np.interp(video_t, wave_t, gt)
    return video_t, bvp, gt


# 功能：对每段视频计算平均HR
# 注释：1. 大约有5个文件时间对不齐需要删除；2. 有的检测不到lmk
def MyEval(HR_pr, HR_rel):
    HR_pr = np.array(HR_pr).reshape(-1)
    HR_rel = np.array(HR_rel).reshape(-1)
    temp = HR_pr - HR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp))/len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2))/len(temp))
    mer = np.mean(np.abs(temp) / HR_rel)
    p = np.sum((HR_pr - np.mean(HR_pr))*(HR_rel - np.mean(HR_rel))) / (
                0.01 + np.linalg.norm(HR_pr - np.mean(HR_pr), ord=2) * np.linalg.norm(HR_rel - np.mean(HR_rel), ord=2))
    
    result = 'me: %.4f\n' % me + \
             'std: %.4f\n' % std + \
             'mae: %.4f\n' % mae + \
             'rmse: %.4f\n' % rmse + \
             'mer: %.4f\n' % mer + \
             'p: %.4f\n' % p
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    folder_path = './results'
    file_path = os.path.join(folder_path, 'pure.txt')
  
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
  
    with open(file_path, 'a') as file:
        file.write('current_time: ' + current_time + '\n' + result + '\n')

    print('| me: %.4f' % me,
          '| std: %.4f' % std,
          '| mae: %.4f' % mae,
          '| rmse: %.4f' % rmse,
          '| mer: %.4f' % mer,
          '| p: %.4f' % p
          )
    return me, std, mae, rmse, mer, p


def main(args):
    gt_name = 'Label_CSI/HR.mat'
    frames_num = 224
    gt_av = []
    pr_av = []
    gt_ps = []
    pr_ps = []
    # Idex_files = r'/home/haolu/Data/VIPL_Index/VIPL_Map51250_Test'
    Idex_files = r'/STW-MAE/data/Map50/PURE_Test'
    pr_path = r'./finetune/finetune_PURE/HR_pr.mat'
    rel_path = r'./finetune/finetune_PURE/HR_rel.mat'
    pr = scio.loadmat(pr_path)['HR_pr']
    pr = np.array(pr.astype('float32')).reshape(-1)
    rel = scio.loadmat(rel_path)['HR_rel']
    rel = np.array(rel.astype('float32')).reshape(-1)
    # files_list = os.listdir(Idex_files)
    files_list = sorted(os.listdir(Idex_files))
    temp = scio.loadmat(os.path.join(Idex_files, files_list[0]))
    lastPath = str(temp['Path'][0])  # 例如：'/STW-MAE/data/ST/PURE/01-01'
    pr_temp = []
    gt_temp = []
    for HR_index in trange(pr.size-1):
        temp = scio.loadmat(os.path.join(Idex_files, files_list[HR_index]))
        nowPath = str(temp['Path'][0])  # 例如：'/STW-MAE/data/ST/PURE/01-01'
        Step_Index = int(temp['Step_Index'][0,0])  # 例如：0
        a = pr[HR_index]
        b = rel[HR_index]

        if lastPath != nowPath:
            # print(')************')
            if pr_temp is None:
                # print('nowPath:', nowPath)
                # print('lastPath:', lastPath)
                pr_temp = []
                gt_temp = []
            else:
                # print('diff_gt', np.array(gt_temp[1:]) - np.array(gt_temp[:-1]))
                # print('diff', np.array(pr_temp[1:]) - np.array(pr_temp[:-1]))
                # print('aaa', np.array(pr_temp)-np.array(gt_temp))
                # print('gt_temp', np.mean(np.array(pr_temp)-np.array(gt_temp)))
                pr_av.append(np.nanmean(pr_temp))
                gt_av.append(np.nanmean(gt_temp))

                # print('len(gt_ps):', len(gt_ps))
                # print('gt_temp:', gt_temp)
                # print('pr_temp:', pr_temp)
                gt_ps.append(gt_temp)
                pr_ps.append(pr_temp)

                pr_temp = []
                gt_temp = []
                
        else:
            gt_path = os.path.join(nowPath, gt_name)  # 例如：'/STW-MAE/data/ST/PURE/01-01/Label_CSI/HR.mat'
            
            subject_name = gt_path.split('/')[-3]  # 例如：'01-01'
            bvp_gt_path = os.path.join('/STW-MAE/data/raw/PURE', subject_name)  # 例如：'/STW-MAE/data/raw/PURE/01-01'
            bvp_gt_name = subject_name+'.json'  # 例如："01-01.json"
            mata_data_file_path = os.path.join(bvp_gt_path, bvp_gt_name)  # 例如：'/STW-MAE/data/raw/PURE/01-01/01-01.json'
            video_t, bvp, gt = read_ground_truth(mata_data_file_path)
            
            # gt = scio.loadmat(gt_path)['HR']
            gt = np.array(gt.astype('float32')).reshape(-1)
            gt = np.nanmean(gt[Step_Index:Step_Index + frames_num])
            gt = gt.astype('float32')
            pr_temp.append(pr[HR_index])
            gt_temp.append(rel[HR_index])
            
        lastPath = nowPath

    # io.savemat(args.output_dir+'/gt_ps.mat', {'HR': gt_ps})
    # io.savemat(args.output_dir+'/pr_ps.mat', {'HR': pr_ps})
    io.savemat(args.output_dir+'/HR_rel.mat', {'HR': gt_av})
    io.savemat(args.output_dir+'/HR_pr.mat', {'HR': pr_av})
    MyEval(gt_av, pr_av)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
