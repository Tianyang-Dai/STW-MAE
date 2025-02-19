import sys
import re
import os
import shutil
import scipy.io as io
import xlrd
import math
import csv
import cv2
import numpy as np
from math import *
from scipy import signal
import scipy.io as scio
from scipy import interpolate
from scipy import signal
import argparse
from tqdm import trange


def get_args_parser():
    parser = argparse.ArgumentParser('STW-MAE data pre-processing', add_help=False)
    parser.add_argument('--STMap_channels', type=str, default='RGB')
    parser.add_argument('--STMap_augmentation', type=str, default='CHROM')
    parser.add_argument('--STMap_name', type=str, default='STMap_RGB_Align_CSI_')

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
        w_step = int(w / 5)
        h_step = int(h / 5)
        for w_index in range(5):
            for h_index in range(5):
                temp = face_crop[h_index * h_step: (h_index + 1) * h_step, w_index * w_step:(w_index + 1) * w_step, :]
                temp1 = np.mean(np.mean(temp, axis=0), axis=0)
                Value.append(temp1)
    return np.array(Value)


def choose_windows(name='Hamming', N=20):
    if name == 'Hamming':  # 汉明窗
        window = np.array([0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Hanning':  # 汉宁窗
        window = np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) for n in range(N)])
    elif name == 'Rect':  # 矩形窗
        window = np.ones(N)
    return window


def CHROM(STMap_CSI):
    LPF = 0.7  # 低截止频率(Hz)，40bpm(~0.667Hz)
    HPF = 2.5  # 高截止频率(Hz)，240bpm(~4.0Hz)
    WinSec = 1.6
    NyquistF = 15
    FS = 30
    FN = STMap_CSI.shape[0]
    B, A = signal.butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass')
    WinL = int(WinSec * FS)
    if (WinL % 2):
        WinL = WinL + 1
    if WinL <= 18:
        WinL = 20
    NWin = int((FN - WinL / 2) / (WinL / 2))
    S = np.zeros(FN)
    WinS = 0
    WinM = WinS + WinL / 2
    WinE = WinS + WinL
    # T = np.linspace(0, FN, FN)
    BGRNorm = np.zeros((WinL, 3))
    for i in range(NWin):
        # TWin = T[WinS:WinE, :]
        for j in range(3):
            BGRBase = np.nanmean(STMap_CSI[WinS:WinE, j])
            BGRNorm[:, j] = STMap_CSI[WinS:WinE, j]/(BGRBase+0.0001) - 1
        Xs = 3*BGRNorm[:, 2] - 2*BGRNorm[:, 1]
        Ys = 1.5*BGRNorm[:, 2] + BGRNorm[:, 1] - 1.5*BGRNorm[:, 0]

        Xf = signal.filtfilt(B, A, np.squeeze(Xs))
        Yf = signal.filtfilt(B, A, np.squeeze(Ys))

        Alpha = np.nanstd(Xf)/np.nanstd(Yf)
        SWin = Xf - Alpha*Yf
        SWin = choose_windows(name='Hanning', N=WinL)*SWin
        if i == 0:
            S[WinS:WinE] = SWin
            # TX[WinS:WinE] = TWin
        else:
            S[WinS: WinM - 1] = S[WinS: WinM - 1] + SWin[0: int(WinL/2) - 1]
            S[WinM: WinE] = SWin[int(WinL/2):]
            # TX[WinM: WinE] = TWin[WinL/2 + 1:]
        WinS = int(WinM)
        WinM = int(WinS + WinL / 2)
        WinE = int(WinS + WinL)
    return S


def POS(STMap_CSI):
    LPF = 0.7  # 低截止频率(Hz)，40bpm(~0.667Hz)
    HPF = 2.5  # 高截止频率(Hz)，240bpm(~4.0Hz)
    WinSec = 1.6
    NyquistF = 15
    FS = 30
    N = STMap_CSI.shape[0]
    l = int(WinSec * FS)
    H = np.zeros(N)
    Cn = np.zeros((3, l))
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    for n in range(N-1):
        m = n - l
        if m >= 0:
            Cn[0, :] = STMap_CSI[m:n, 2]/np.nanmean(STMap_CSI[m:n, 2])
            Cn[1, :] = STMap_CSI[m:n, 1]/np.nanmean(STMap_CSI[m:n, 1])
            Cn[2, :] = STMap_CSI[m:n, 0]/np.nanmean(STMap_CSI[m:n, 0])
            S = np.dot(P, Cn)
            h = S[0, :] + ((np.nanstd(S[0, :])/np.nanstd(S[1, :]))*S[1, :])
            H[m: n] = H[m: n] + (h - np.nanmean(h))
    return H


def mySTMap(imglist_root, lmks=[], Time=[], STMap_augmentation='Original'):
    # 例如：imglist_root='/STW-MAE/data/raw/PURE/01-01/01-01'
    # b, a = signal.butter(5, 0.12 / (30 / 2), 'highpass')
    b, a = signal.butter(5, [0.5 / (30 / 2), 3 / (30 / 2)], 'bandpass')
    img_list = os.listdir(imglist_root)
    z = 0
    STMap = []
    
    Time_elements = len(img_list)
    # Time = [i * 33.3333 for i in range(Time_elements)]
    Time = [i * 1 for i in range(Time_elements)]
    # 例如，Time=[0,1,...,28,29]
    
    for i in trange(len(img_list)):
        imgPath_sub = img_list[i]
        now_path = os.path.join(imglist_root, imgPath_sub)
        img = cv2.imread(now_path)
        Value = getValue(img, lmk=lmks[z])
        if np.isnan(Value).any():
            Value[:, :] = 100
        STMap.append(Value)
        z = z + 1
    STMap = np.array(STMap)
    
    # CSI_Time = np.arange(0, Time[-1], 33.3333)
    CSI_Time = np.arange(0, Time[-1], 1)
    STMap_CSI = np.zeros((len(CSI_Time), STMap.shape[1], STMap.shape[2]))
    
    for c in range(STMap.shape[2]):
        for w in range(STMap.shape[1]):
            if STMap_augmentation=='Filtered':
                STMap[:, w, c] = signal.filtfilt(b, a, np.squeeze(STMap[:, w, c]+0.01))
            t = interpolate.splrep(Time, STMap[:, w, c])
            STMap_CSI[:, w, c] = interpolate.splev(CSI_Time, t)
    
    if STMap_augmentation=='CHROM':
        for w in range(STMap.shape[1]):
            STMap_CSI[:, w, 0] = np.squeeze(CHROM(STMap_CSI[:, w, :]))
    elif STMap_augmentation=='POS':
        for w in range(STMap.shape[1]):
            STMap_CSI[:, w, 0] = np.squeeze(POS(STMap_CSI[:, w, :]))
    
    for c in range(STMap.shape[2]):
        for w in range(STMap.shape[1]):
            STMap_CSI[:, w, c] = 255 * ((STMap_CSI[:, w, c] - np.nanmin(STMap_CSI[:, w, c])) / (
                    0.001 + np.nanmax(STMap_CSI[:, w, c]) - np.nanmin(STMap_CSI[:, w, c])))
            
    STMap_CSI = np.swapaxes(STMap_CSI, 0, 1)
    STMap_CSI = np.rint(STMap_CSI)
    STMap_CSI = np.array(STMap_CSI, dtype='uint8')
    
    return STMap_CSI


def main(args):
    preprocessedRoot = r'/STW-MAE/data/preprocessed/PURE/'  # 预处理根目录
    rawRoot = r'/STW-MAE/data/raw/PURE/'  # 原始根目录
    MapRoot = r'/STW-MAE/data/Map/PURE/'  # Map根目录
    
    STMap_name = args.STMap_name + args.STMap_augmentation + '.png'  # STMap名称，例如：'STMap_RGB_Align_CSI_CHROM.png'
    
    sessions = os.listdir(preprocessedRoot)  # 会话
    sessions.sort()
    
    for session in sessions:  # 例如：session='01-01'
        print('STMap augmentation:', args.STMap_augmentation)
        
        preprocessed_path = os.path.join(preprocessedRoot, session)  # 预处理路径，例如：'/STW-MAE/data/preprocessed/PURE/01-01'
        raw_path = os.path.join(rawRoot, session)  # 原始路径，例如：'/STW-MAE/data/raw/PURE/01-01'
        Map_path = os.path.join(MapRoot, session)  # STMap路径，例如：'/STW-MAE/data/Map/PURE/01-01'
        lmk_path = os.path.join(preprocessed_path, session+'.csv')  # 关键点路径，例如：'/STW-MAE/data/preprocessed/PURE/01-01/01-01.csv'
        image_path = os.path.join(raw_path, session)  # RGB路径，例如：'/STW-MAE/data/raw/PURE/01-01/01-01'
        STMap_path = os.path.join(Map_path, 'STMap')  # STMap路径，例如：'/STW-MAE/data/Map/PURE/01-01/STMap'
        
        lmks = []  # 形状为(行数(1+帧数), 列数(4+68点*2))，例如：(2027, 141)
        with open(os.path.join(lmk_path), "r") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lmks.append(line)
        lmks = np.array(lmks[1:], dtype='float32')[:, 5:]  # 删除第一行和前四列，形状为(行数(帧数), 列数(68点*2))，例如：(2026, 136)
        
        if not os.path.exists(STMap_path):
            os.makedirs(STMap_path)
            
        STMap = mySTMap(image_path, lmks=lmks, Time=[], STMap_augmentation=args.STMap_augmentation)
        cv2.imwrite(os.path.join(STMap_path, STMap_name), STMap, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
        print('session:', session)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
