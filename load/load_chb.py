import glob
import os
import numpy as np
import mne
from mne.filter import filter_data
from scipy.stats import skew, kurtosis
from tqdm import tqdm
import re
import pywt

def extract_band_power_features_with_pooling(X, sfreq, bands):
    N, C, T = X.shape
    print('N,C,T', N, C, T)
    band_power_pooled = []
    for _, (fmin, fmax) in bands.items():
        X_band = np.zeros_like(X)
        for c in range(C):
            X_band[:, c, :] = filter_data(X[:, c, :], sfreq, l_freq=fmin, h_freq=fmax, verbose='error')
        power = np.mean(X_band ** 2, axis=2)  # (N, C)
        avg_power = np.mean(power, axis=1)    # (N,)
        band_power_pooled.append(avg_power)
    band_features = np.stack(band_power_pooled, axis=1)  # (N, B)

    X_flat = X.reshape(N, -1)
    mean = np.mean(X_flat, axis=1)
    std = np.std(X_flat, axis=1)
    skw = skew(X_flat, axis=1)
    krt = kurtosis(X_flat, axis=1)
    stat_features = np.stack([mean, std, skw, krt], axis=1)  # (N, 4)

    # features = np.concatenate([band_features, stat_features], axis=1)  # (N, 8)
    features = band_features
    return features

# def extract_band_power_features_with_pooling(X, sfreq, bands):
#     N, C, T = X.shape
#     print('N, C, T:', N, C, T)

#     energy_all = []
#     var_all = []

#     for _, (fmin, fmax) in bands.items():
#         X_band = np.zeros_like(X)
#         for c in range(C):
#             X_band[:, c, :] = filter_data(X[:, c, :], sfreq, l_freq=fmin, h_freq=fmax, verbose='error')

#         # Energy：对时间求和
#         energy = np.sum(X_band ** 2, axis=2)    # shape: (N, C)
#         avg_energy = np.mean(energy, axis=1)    # shape: (N,)

#         # Variance：对时间求方差
#         var = np.var(X_band, axis=2)            # shape: (N, C)
#         avg_var = np.mean(var, axis=1)          # shape: (N,)

#         energy_all.append(avg_energy)
#         var_all.append(avg_var)

#     # 拼接成最终特征向量 (N, B*2)
#     energy_all = np.stack(energy_all, axis=1)  # (N, 5)
#     var_all = np.stack(var_all, axis=1)        # (N, 5)
#     features = np.concatenate([energy_all, var_all], axis=1)  # (N, 10)

#     return features


def extract_wavelet_energy_variance_features(X, wavelet='db4', level=5):
    """
    提取小波系数在每个分解层的 energy 和 variance 特征
    X: EEG data, shape (N, C, T)
    return: features of shape (N, 2 * (level + 1))  → (N, 12) for level=5
    """
    N, C, T = X.shape
    print('N, C, T:', N, C, T)

    energy_all = []
    var_all = []

    for i in range(N):  # 每个 trial
        trial_energy = []
        trial_var = []

        for c in range(C):  # 每个通道
            signal = X[i, c, :]
            coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)  # coeffs = [A5, D5, D4, ..., D1]

            for coeff in coeffs:
                trial_energy.append(np.sum(np.square(coeff)))  # 能量
                trial_var.append(np.var(coeff))                # 方差

        # 对当前样本所有通道进行平均（每一层系数都有 C 份，取平均）
        n_coeff = len(trial_energy) // C  # 6层（A5 + D1~D5）
        avg_energy = [np.mean(trial_energy[i::n_coeff]) for i in range(n_coeff)]
        avg_var = [np.mean(trial_var[i::n_coeff]) for i in range(n_coeff)]

        energy_all.append(avg_energy)
        var_all.append(avg_var)

    energy_all = np.array(energy_all)  # shape: (N, 6)
    var_all = np.array(var_all)        # shape: (N, 6)

    features = np.concatenate([energy_all, var_all], axis=1)  # shape: (N, 12)

    return features

def preprocess_and_extract_features_mne_with_timestamps(file_name, bands):
    raw = mne.io.read_raw_edf(file_name, preload=True, verbose='error')
    raw.filter(1., 50., fir_design='firwin', verbose='error')
    raw.pick_types(meg=False, eeg=True, eog=False)
    sfreq = raw.info['sfreq']

    window_length_sec = 3
    window_samples = int(window_length_sec * sfreq)
    data = raw.get_data()  # shape: (C, T)
    C, T = data.shape

    segments, timestamps = [], []
    for start in range(0, T - window_samples + 1, window_samples):
        end = start + window_samples
        segment = data[:, start:end]
        segments.append(segment)
        timestamps.append(raw.times[start])  # 起始时间

    segments = np.array(segments)  # (N, C, T)
    features = extract_band_power_features_with_pooling(segments, sfreq, bands)
    # features = extract_wavelet_energy_variance_features(segments)
    timestamps = np.array(timestamps)
    return timestamps, features

def extractTarget(summary_file_path, edf_file_path):
    edf_file_name = os.path.basename(edf_file_path)
    seizure_start_time = None
    seizure_end_time = None
    with open(summary_file_path, 'r') as file:
        lines = file.readlines()
    found = False
    for line in lines:
        if "File Name: " + edf_file_name in line:
            found = True
        if found:
            if "Number of Seizures in File: 0" in line:
                return None, None
            match_start = re.search(r"Seizure\s+\d*\s*Start Time:\s*(\d+)", line)
            if match_start:
                seizure_start_time = int(match_start.group(1))

            match_end = re.search(r"Seizure\s+\d*\s*End Time:\s*(\d+)", line)
            if match_end:
                seizure_end_time = int(match_end.group(1))
                break  # 只取第一个 seizure 的开始和结束时间
    return seizure_start_time, seizure_end_time

def extract_data_and_labels(edf_file_path, summary_file_path, bands):
    timestamps, X = preprocess_and_extract_features_mne_with_timestamps(edf_file_path, bands)
    seizure_start_time, seizure_end_time = extractTarget(summary_file_path, edf_file_path)
    print('癫痫时刻', seizure_start_time, seizure_end_time)
    if seizure_start_time is None or seizure_end_time is None:
        y = np.zeros(len(X), dtype=int)
    else:
        y = np.array([1 if seizure_start_time <= t <= seizure_end_time else 0 for t in timestamps])
    return X, y

def load_data(subject_id, base_path):
    edf_file_paths = sorted(glob.glob(os.path.join(base_path, f"chb{subject_id:02d}/*.edf")))
    summary_file_path = os.path.join(base_path, f"chb{subject_id:02d}/chb{subject_id:02d}-summary.txt")
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 50)}

    all_X, all_y = [], []
    for edf_file in tqdm(edf_file_paths, desc=f"Processing Subject {subject_id}"):
        X, y = extract_data_and_labels(edf_file, summary_file_path, bands)
        all_X.append(X)
        all_y.append(y)

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # 类别平衡：下采样非发作类
    idx_pos = np.where(y_all == 1)[0]
    idx_neg = np.where(y_all == 0)[0]
    np.random.seed(42)
    idx_neg_sampled = np.random.choice(idx_neg, size=len(idx_pos), replace=False)
    idx_final = np.concatenate([idx_pos, idx_neg_sampled])
    np.random.shuffle(idx_final)

    X_all = X_all[idx_final]
    y_all = y_all[idx_final]

    np.savez(f'data/chb-mit/chb_data_balanced_{subject_id}.npz', X=X_all, y=y_all)
    print(f"Saved balanced dataset: {X_all.shape}, {y_all.shape}")
    return X_all, y_all

if __name__ == '__main__':
    subject_id = 10

    # 处理
    base_path = "/data1/labram_data/chb-mit/"
    X_all, y_all = load_data(subject_id, base_path)

    # # 读取
    # data = np.load(f'chb_data_balanced_{subject_id}.npz')
    # X_all = data['X']
    # y_all = data['y']

    print("Final X_all shape:", X_all.shape)
    print("Final y_all shape:", y_all.shape)

    unique, counts = np.unique(y_all, return_counts=True)
    print("Label distribution:")
    for label, count in zip(unique, counts):
        print(f"Label {label}: {count}")
