import sys
import re
import os
import shutil
import xlrd
import math
from math import *
import csv
import cv2
import numpy as np
import scipy.io as io
from scipy import signal
import scipy.io as scio
from scipy import interpolate
from scipy import signal
import argparse
from tqdm import trange
from denoising import *
import matplotlib.pyplot as plt
import pywt
from PIL import Image


def get_args_parser():
    parser = argparse.ArgumentParser('STW-MAE data pre-processing', add_help=False)
    parser.add_argument('--WaveletMap_channels', type=str, default='RGB')
    parser.add_argument('--WaveletMap_name', type=str, default='WaveletMap_RGB_Align_CSI')

    return parser


def PointRotate(angle, valuex, valuey, pointx, pointy):
    valuex = np.array(valuex)
    valuey = np.array(valuey)
    Rotatex = (valuex - pointx) * math.cos(angle) - (valuey - pointy) * math.sin(angle) + pointx
    Rotatey = (valuex - pointx) * math.sin(angle) + (valuey - pointy) * math.cos(angle) + pointy
    return Rotatex, Rotatey


def getValue(img, lmk=[], type=2, lmk_type=2, channels='RGB'):
    Value = []
    # 1. 三点对齐；2. 两点对齐
    # 1. 81点；2. 68点
    h, w, c = img.shape
    if channels == 'YUV':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    if type == 1:  # 三点对齐
        w_step = int(w / 5)
        h_step = int(h / 5)
        for w_index in range(5):
            for h_index in range(5):
                temp = img[h_index * h_step: (h_index + 1) * h_step, w_index * w_step:(w_index + 1) * w_step, :]
                temp1 = np.nanmean(np.nanmean(temp, axis=0), axis=0)
                Value.append(temp1)
                
    elif type == 2:  # 两点对齐（默认）
        # 形状x_0,x_1,...,x_67,y_0,y_1,...,y_67改为x_0,y_0,x_1,y_1,...,x_67,y_67
        lmk = np.array(lmk, np.float32).reshape(1, -1)
        new_lmk = np.zeros((1, 136))
        for i in range(68):
            new_lmk[0, 2*i] = lmk[0, i]
            new_lmk[0, 2*i+1] = lmk[0, i+68]
        lmk = new_lmk
        
        lmk = np.array(lmk, np.float32).reshape(-1, 2)
        min_p = np.min(lmk, 0)
        max_p = np.max(lmk, 0)
        min_p = np.maximum(min_p, 0)
        max_p = np.minimum(max_p, [w - 1, h-1])
        
        if lmk_type == 1:  # 81点
            left_eye = lmk[0:8]
            right_eye = lmk[9:17]
            left = np.array([lmk[60], lmk[62], lmk[65]])
            right = np.array([lmk[61], lmk[63], lmk[73]])
        else:  # 68点（默认）
            left_eye = lmk[36:41]
            right_eye = lmk[42:47]
            left = np.array([lmk[0], lmk[1], lmk[2]])
            right = np.array([lmk[14], lmk[15], lmk[16]])
            
        left_eye = np.nanmean(left_eye, 0)
        right_eye = np.nanmean(right_eye, 0)
        left = np.nanmean(left, 0)
        right = np.nanmean(right, 0)
        top = max((left[1] + right[1])/2 - 0.5*(max_p[1] - (left[1] + right[1])/2), 0)
        rotate_angular = math.atan((right_eye[1] - left_eye[1]) / (0.00001+right_eye[0] - left_eye[0])) * (180 / math.pi)
      
        cent_point = [w/2, h/2]
        matRotation = cv2.getRotationMatrix2D((w/2, h/2), rotate_angular, 1)
        face_rotate = cv2.warpAffine(img, matRotation, (w, h))
        left[0], left[1] = PointRotate(math.radians(rotate_angular), left[0], left[1], cent_point[0], cent_point[1])
        right[0], right[1] = PointRotate(math.radians(rotate_angular), right[0], right[1], cent_point[0], cent_point[1])
      
        face_crop = face_rotate[int(top):int(max_p[1]), int(left[0]):int(right[0]), :]
        # cv2.imshow('a', face_crop)
        # cv2.waitKey(0)
        h, w, c = face_crop.shape
        w_step = int(w / 1)
        h_step = int(h / 1)
        for w_index in range(1):
            for h_index in range(1):
                temp = face_crop[h_index * h_step: (h_index + 1) * h_step, w_index * w_step:(w_index + 1) * w_step, :]
                temp1 = np.mean(np.mean(temp, axis=0), axis=0)
                Value.append(temp1)
    return np.array(Value)


# totalscal小波的尺度，对应频谱分析结果也就是分析几个（totalscal-1）频谱
def TimeFrequencyCWT(data, fs, totalscal, wavelet='cgau8'):
    # 采样数据的时间维度
    t = np.arange(data.shape[0])/fs
    # 中心频率
    wcf = pywt.central_frequency(wavelet=wavelet)
    # 计算对应频率的小波尺度
    cparam = 2 * wcf * totalscal
    scales = cparam/np.arange(totalscal, 1, -1)
    # 连续小波变换
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavelet, 1.0/fs)
    
    # plt.figure(figsize=(8, 4))
    # plt.subplot(211)
    # plt.plot(t, data)
    # plt.xlabel(u"time(s)")
    # plt.title(u"Time spectrum")
    # plt.subplot(212)
    # plt.contourf(t, frequencies, abs(cwtmatr))
    # plt.ylabel(u"freq(Hz)")
    # plt.xlabel(u"time(s)")
    # plt.subplots_adjust(hspace=0.4)
    # plt.show()
    
    # 计算幅度矩阵
    WaveletMap = np.transpose(np.abs(cwtmatr))  # 形状为(帧数-1, totalscal-1)，例如(2025, 25)

    # 归一化WaveletMap到[0, 1]区间
    # normalized_wavelet_map = (WaveletMap - np.min(WaveletMap)) / (np.max(WaveletMap) - np.min(WaveletMap))
    # 缩放到[0, 255]区间
    # scaled_wavelet_map = (normalized_wavelet_map * 255).astype(np.uint8)
    
    return WaveletMap


def myWaveletMap(imglist_root, lmks=[], Time=[]):
    # 例如imglist_root='/STW-MAE/data/raw/PURE/01-01/01-01'
    # b, a = signal.butter(5, 0.12 / (30 / 2), 'highpass')
    b, a = signal.butter(5, [0.5 / (30 / 2), 3 / (30 / 2)], 'bandpass')
    img_list = os.listdir(imglist_root)
    z = 0
    WaveletMap = []
    
    Time_elements = len(img_list)
    # Time = [i * 33.3333 for i in range(Time_elements)]
    Time = [i * 1 for i in range(Time_elements)]
    # 例如Time=[0,1,...,28,29]
    
    for i in trange(len(img_list)):
        imgPath_sub = img_list[i]
        now_path = os.path.join(imglist_root, imgPath_sub)
        img = cv2.imread(now_path)
        Value = getValue(img, lmk=lmks[z])
        if np.isnan(Value).any():
            Value[:, :] = 100
        WaveletMap.append(Value)
        z = z + 1
    WaveletMap = np.array(WaveletMap)  # 形状为(帧数, 1, 通道)，例如(2026, 1, 3)
    
    # CSI_Time = np.arange(0, Time[-1], 33.3333)
    CSI_Time = np.arange(0, Time[-1], 1)
    WaveletMap_CSI = np.zeros((len(CSI_Time), WaveletMap.shape[1], WaveletMap.shape[2]))  # 形状为(帧数-1, 1, 通道)，例如(2025, 1, 3)
    
    h_WaveletMap = 25
  
    for c in range(WaveletMap.shape[2]):
        for w in range(WaveletMap.shape[1]):
            WaveletMap[:, w, c] = signal.filtfilt(b, a, np.squeeze(WaveletMap[:, w, c]+0.01))
            WaveletMap[:, w, c] = ADLF_ti(WaveletMap[:, w, c].flatten(), level=h_WaveletMap+1)  # 自适应调整步长的晚期融合的小波平移不变消噪
            t = interpolate.splrep(Time, WaveletMap[:, w, c])
            WaveletMap_CSI[:, w, c] = interpolate.splev(CSI_Time, t)
    
    temp_WaveletMap_CSI = np.zeros((len(CSI_Time), h_WaveletMap, WaveletMap.shape[2]))  # 形状为(帧数-1, h_WaveletMap, 通道)，例如(2025, 25, 3)
    for c in range(WaveletMap_CSI.shape[2]):
        temp_WaveletMap_CSI[:, :, c] = TimeFrequencyCWT(WaveletMap_CSI[:, :, c].flatten(), fs=30, totalscal=h_WaveletMap+1, wavelet='cgau8')
    WaveletMap_CSI = temp_WaveletMap_CSI  # 形状为(帧数-1, h_WaveletMap, 通道)，例如(2025, 25, 3)
    
    for c in range(WaveletMap_CSI.shape[2]):
        for w in range(WaveletMap_CSI.shape[1]):
            WaveletMap_CSI[:, w, c] = 255 * ((WaveletMap_CSI[:, w, c] - np.nanmin(WaveletMap_CSI[:, w, c])) / (
                    0.001 + np.nanmax(WaveletMap_CSI[:, w, c]) - np.nanmin(WaveletMap_CSI[:, w, c])))
            
    WaveletMap_CSI = np.swapaxes(WaveletMap_CSI, 0, 1)
    WaveletMap_CSI = np.rint(WaveletMap_CSI)
    WaveletMap_CSI = np.array(WaveletMap_CSI, dtype='uint8')
    
    return WaveletMap_CSI


def main(args):
    preprocessedRoot = r'/STW-MAE/data/preprocessed/PURE/'  # 预处理根目录
    rawRoot = r'/STW-MAE/data/raw/PURE/'  # 原始根目录
    MapRoot = r'/STW-MAE/data/Map/PURE/'  # Map根目录
    
    WaveletMap_name = args.WaveletMap_name + '.png'  # WaveletMap名称，例如'WaveletMap_RGB_Align_CSI.png'
    
    sessions = os.listdir(preprocessedRoot)  # 会话
    sessions.sort()
    
    for session in sessions:  # 例如session='01-01'
        preprocessed_path = os.path.join(preprocessedRoot, session)  # 预处理路径，例如'/STW-MAE/data/preprocessed/PURE/01-01'
        raw_path = os.path.join(rawRoot, session)  # 原始路径，例如'/STW-MAE/data/raw/PURE/01-01'
        Map_path = os.path.join(MapRoot, session)  # WaveletMap路径，例如'/STW-MAE/data/Map/PURE/01-01'
        lmk_path = os.path.join(preprocessed_path, session+'.csv')  # 关键点路径，例如'/STW-MAE/data/preprocessed/PURE/01-01/01-01.csv'
        image_path = os.path.join(raw_path, session)  # RGB路径，例如'/STW-MAE/data/raw/PURE/01-01/01-01'
        WaveletMap_path = os.path.join(Map_path, 'WaveletMap')  # WaveletMap路径，例如'/STW-MAE/data/Map/PURE/01-01/WaveletMap'
        
        lmks = []  # 形状为(行数(1+帧数), 列数(4+68点*2))，例如(2027, 141)
        with open(os.path.join(lmk_path), "r") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lmks.append(line)
        lmks = np.array(lmks[1:], dtype='float32')[:, 5:]  # 删除第一行和前四列，形状为(行数(帧数), 列数(68点*2))，例如(2026, 136)
        
        if not os.path.exists(WaveletMap_path):
            os.makedirs(WaveletMap_path)
            
        WaveletMap = myWaveletMap(image_path, lmks=lmks, Time=[])
        cv2.imwrite(os.path.join(WaveletMap_path, WaveletMap_name), WaveletMap, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
        print('session:', session)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
