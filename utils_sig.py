import numpy as np
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt


def butter_bandpass(sig, lowcut, highcut, fs, order=2):
    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    
    y = filtfilt(b, a, sig)
    return y


def butter_bandpass_batch(sig_list, lowcut, highcut, fs, order=2):
    y_list = []
    
    for sig in sig_list:
        sig = np.reshape(sig, -1)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, sig)
        y_list.append(y)
    return np.array(y_list)


def hr_fft(sig, fs, harmonics_removal=True):
    sig = sig.reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
    high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
    sig_f_original = sig_f.copy()
    
    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0

    peak_idx, _ = signal.find_peaks(sig_f)
    sort_idx = np.argsort(sig_f[peak_idx])
    sort_idx = sort_idx[::-1]

    peak_idx1 = peak_idx[sort_idx[0]]
    peak_idx2 = peak_idx[sort_idx[1]]

    f_hr1 = peak_idx1 / sig.shape[0] * fs
    hr1 = f_hr1 * 60

    f_hr2 = peak_idx2 / sig.shape[0] * fs
    hr2 = f_hr2 * 60
    if harmonics_removal:
        if np.abs(hr1-2*hr2)<10:
            hr = hr2
        else:
            hr = hr1
    else:
        hr = hr1

    x_hr = np.arange(len(sig))/len(sig)*fs*60
    return hr, sig_f_original, x_hr


def hr_fft_batch(sig_list, fs, harmonics_removal=True):
    hr_list = []
    sig_original_list = []
    x_hr_list = []
    for sig in sig_list:
        sig = sig.reshape(-1)
        sig = sig * signal.windows.hann(sig.shape[0])
        sig_f = np.abs(fft(sig))
        low_idx = np.round(0.6 / fs * sig.shape[0]).astype('int')
        high_idx = np.round(4 / fs * sig.shape[0]).astype('int')
        sig_f_original = sig_f.copy()
        
        sig_f[:low_idx] = 0
        sig_f[high_idx:] = 0

        peak_idx, _ = signal.find_peaks(sig_f)
        sort_idx = np.argsort(sig_f[peak_idx])
        sort_idx = sort_idx[::-1]

        peak_idx1 = peak_idx[sort_idx[0]]
        peak_idx2 = peak_idx[sort_idx[1]]

        f_hr1 = peak_idx1 / sig.shape[0] * fs
        hr1 = f_hr1 * 60

        f_hr2 = peak_idx2 / sig.shape[0] * fs
        hr2 = f_hr2 * 60
        if harmonics_removal:
            if np.abs(hr1-2*hr2)<10:
                hr = hr2
            else:
                hr = hr1
        else:
            hr = hr1

        x_hr = np.arange(len(sig))/len(sig)*fs*60
        hr_list.append(hr)
        sig_original_list.append(sig_f_original)
        x_hr_list.append(x_hr)
    return np.array(hr_list),np.array(sig_original_list),np.array(x_hr_list)


def normalize(x):
    return (x-x.mean())/x.std()
