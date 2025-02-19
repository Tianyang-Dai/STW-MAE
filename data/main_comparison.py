from denoising import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend

# 创建时间轴，从0到1，共计1000个点
time = np.linspace(0, 1, 1000)
# 创建原始信号，包含三个正弦波
original_signal = np.sin(2 * np.pi * 5 * time) + np.sin(2 * np.pi * 7 * time * time) + np.sin(2 * np.pi * 9 * time * time * time)
# 创建含噪信号，原始信号加上高斯噪声
noisy_signal = original_signal + 0.5 * np.random.randn(len(time))

# 使用ti函数对含噪信号进行去噪
denoised_signal_1 = ti(noisy_signal)
# 使用ADLF_ti函数对含噪信号进行去噪
denoised_signal_2 = ADLF_ti(noisy_signal)

plt.figure(figsize=(12, 6))

# 在第一个子图中绘制原始信号
plt.subplot(4, 1, 1)
plt.plot(time, original_signal, label='Original Signal')
plt.legend()

# 在第二个子图中绘制含噪信号
plt.subplot(4, 1, 2)
plt.plot(time, noisy_signal, label='Noisy Signal')
plt.legend()

# 在第三个子图中绘制经过ti方法去噪后的信号
plt.subplot(4, 1, 3)
plt.plot(time, denoised_signal_1, label='Translation Invariant Denoised Signal')
plt.legend()

# 在第四个子图中绘制经过ADLF_ti方法去噪后的信号
plt.subplot(4, 1, 4)
plt.plot(time, denoised_signal_2, label='Late Fusion Translation Invariant Denoised Signal with Adaptive Step Size Adjustment (Ours)')
plt.legend()

plt.tight_layout()
plt.savefig('./data/comparison.png')
plt.show()
