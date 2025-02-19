import pywt
import math
import numpy as np
import matplotlib.pylab as plt
from threshold import sure_shrink, heur_sure, visu_shrink, mini_max
from ops import right_shift, back_shift, get_var


# 获取近似baseline
def get_baseline(data, wavelets_name='sym8', level=5):
    """
    使用小波分解技术获得数据的近似baseline
    data: 待处理的信号数据
    wavelets_name: 使用的小波基函数
    level: 小波分解的级别
    """
    # 初始化指定小波基的对象
    wave = pywt.Wavelet(wavelets_name)
    
    # 对数据进行小波分解
    coeffs = pywt.wavedec(data, wave, level=level)
    
    # 保留分解后的直流系数，其余系数置零
    for i in range(1, len(coeffs)):
        coeffs[i] *= 0
    
    # 通过重构得到近似的baseline信号
    baseline = pywt.waverec(coeffs, wave)
    
    return baseline


# 阈值收缩去噪法
def tsd(data, method='sureshrink', mode='soft', wavelets_name='sym8', level=5):
    """
    使用小波阈值去噪方法对数据进行处理
    - data: 需要处理的数据
    - method: 使用的阈值方法，支持['sureshrink', 'visushrink', 'heursure', 'minmax']
    - mode: 阈值处理的模式，支持['soft', 'hard', 'garotte', 'greater', 'less']
    - wavelets_name: 使用的小波函数名称
    - level: 小波分解的级别
    """
    # 根据method参数选择相应的阈值计算函数
    methods_dict = {'sureshrink': sure_shrink, 'visushrink': visu_shrink, 'heursure': heur_sure, 'minmax': mini_max}
    # 创建小波对象
    wave = pywt.Wavelet(wavelets_name)

    # 初始化数据用于后续处理
    data_ = data[:]

    # 进行单级小波分解，用于计算变差系数
    (cA, cD) = pywt.dwt(data=data_, wavelet=wave)
    var = get_var(cD)

    # 进行多级小波分解
    coeffs = pywt.wavedec(data=data, wavelet=wavelets_name, level=level)

    # 遍历分解后的系数，对除第一级外的系数进行阈值处理
    for idx, coeff in enumerate(coeffs):
        if idx == 0:
            continue
        # 根据选择的阈值方法计算阈值
        thre = methods_dict[method](var, coeff)
        # 对系数进行阈值处理
        coeffs[idx] = pywt.threshold(coeffs[idx], thre, mode=mode)

    # 通过重构系数得到处理后的数据
    thresholded_data = pywt.waverec(coeffs, wavelet=wavelets_name)

    # 确保thresholded_data与data长度相等
    if len(thresholded_data) > len(data):
        # 如果thresholded_data比data长，截断thresholded_data
        thresholded_data = thresholded_data[:len(data)]
    elif len(thresholded_data) < len(data):
        # 如果thresholded_data比data短，填充thresholded_data
        thresholded_data.extend(data[len(thresholded_data):])
    
    return thresholded_data


# 小波平移不变消噪
def ti(data, step=100, method='sureshrink', mode='soft', wavelets_name='sym5', level=5):
    """
    实现小波平移不变去噪处理
    data: 待处理的信号数据
    step: 平移步长
    method: 去噪方法
    mode: 小波去噪的模式
    wavelets_name: 使用的小波函数名称
    level: 小波分解的层次
    """
    # 计算需要处理的数据段数
    num = math.ceil(len(data)/step)
    # 初始化存储最终结果的列表
    final_data = [0]*len(data)
    
    # 遍历每个数据段进行处理
    for i in range(num):
        # 将数据段右移step个位置
        temp_data = right_shift(data, i*step)
        # 对右移后的数据段进行小波去噪处理
        temp_data = tsd(temp_data, method=method, mode=mode, wavelets_name=wavelets_name, level=level)
        # 将处理后的数据段左移step个位置
        temp_data = temp_data.tolist()
        temp_data = back_shift(temp_data, i*step)
        # 将当前数据段处理结果与之前的结果累加
        final_data = list(map(lambda x, y: x+y, final_data, temp_data))

    # 将最终结果除以数据段数，求平均值，以实现平移不变性
    final_data = list(map(lambda x: x/num, final_data))
    
    # 确保final_data与data长度相等
    if len(final_data) > len(data):
        # 如果final_data比data长，截断final_data
        final_data = final_data[:len(data)]
    elif len(final_data) < len(data):
        # 如果final_data比data短，填充final_data
        final_data.extend(data[len(final_data):])

    # 返回处理后的数据
    return final_data


# 自适应调整步长的晚期融合的小波平移不变消噪
def calculate_energy(data, window_size):
    """
    计算数据序列的能量
    通过将数据划分为固定大小的窗口，并计算每个窗口内数据点的平方和，来度量能量
    这种方法常用于信号处理或音频分析中，计算每个时间窗口的能量可以帮助理解数据的波动或活动水平
    data: 包含需要计算能量的数据序列
    window_size: 指定滑动窗口的大小
    """
    # 初始化用于存储能量的列表
    energy = []
    # 遍历数据序列，为每个窗口计算能量
    for i in range(len(data) - window_size + 1):
        # 提取当前窗口的数据
        window = data[i:i + window_size]
        # 计算窗口内数据的平方和，即能量
        energy.append(np.sum(np.abs(window) ** 2))
    # 将能量列表转换为numpy数组并返回
    return np.array(energy)


def adaptive_step(data, window_size=100, min_step=10, max_step=100):
    """
    根据数据的能谱特征自适应调整步长
    此函数计算给定数据的能谱特征，并根据这些特征动态调整步长，以在搜索中找到最佳的平衡点，
    既不过度细化搜索导致计算量过大，也不过度粗化导致搜索精度不足
    data: 需要分析的数据序列
    window_size: 计算能谱特征的窗口大小
    min_step: 步长的最小值
    max_step: 步长的最大值
    """
    # 计算数据的能谱特征
    energy = calculate_energy(data, window_size)
    # 根据能谱特征调整步长，确保步长在最小值和最大值之间
    step = np.maximum(min_step, np.minimum(max_step, window_size / (energy + 1)))
    # 将计算得到的浮点数步长转换为整数类型返回
    return step.astype(int)


def late_fusion(data, num=4, method='sureshrink', mode='soft', wavelets_name='sym5', level=5):
    """
    实现多尺度时间序列分析的晚期融合算法
    data: 时间序列数据数组
    num: 要应用的变换数量
    method: 变换方法的名称，可选方法包括['sureshrink', 'visushrink', 'heursure', 'minmax']
    mode: 变换的模式
    wavelets_name: 小波变换使用的母小波名称
    level: 小波分解的级别
    """
    
    # 定义可用的变换方法列表
    methods = ['sureshrink', 'visushrink', 'heursure', 'minmax']
    # 初始化一个与输入数据相同形状的零数组，用于存储融合后的数据
    late_fusion_data = np.zeros_like(data)
    
    # 对每个指定的变换方法进行处理
    for i in range(num):
        # 从方法列表中选择当前的变换方法
        method = methods[i]
        # 对输入数据应用小波变换，并根据当前方法更新数据数组
        temp_data = tsd(data, method=method, mode=mode, wavelets_name=wavelets_name, level=level)
        # 累加每个变换结果到融合数据数组中
        late_fusion_data += temp_data
    
    # 将融合数据数组的值平均，以得到最终的融合结果
    late_fusion_data /= float(num)
    
    # 返回经过晚期融合处理后的数据
    return late_fusion_data


def ADLF_ti(data, method='sureshrink', mode='soft', wavelets_name='sym5', level=5):
    # 计算自适应步长
    step = adaptive_step(data)
    
    # 循环平移
    final_data = [0] * len(data)
    num = max(step)  # 取最大步长作为循环次数的估计
    for i in range(num):
        temp_data = right_shift(data, i * step[i % len(step)])
        temp_data = late_fusion(temp_data, num=4, method=method, mode=mode, wavelets_name=wavelets_name, level=level)
        temp_data = temp_data.tolist()
        temp_data = back_shift(temp_data, i * step[i % len(step)])
        final_data = [x + y for x, y in zip(final_data, temp_data)]

    final_data = [x / num for x in final_data]

    return final_data
