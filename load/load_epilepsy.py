import os
import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.filter import filter_data
from sklearn.model_selection import StratifiedKFold
from scipy.linalg import fractional_matrix_power
from scipy.spatial.distance import euclidean
from scipy.stats import skew, kurtosis
from scipy.signal import welch

def EA(x):
    new_x = np.zeros_like(x) #(N,C,T)
    cov = np.zeros((x.shape[0],x.shape[1],x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov,0)
    sqrtRefEA = fractional_matrix_power(refEA,-0.5)
    new_x = np.matmul(sqrtRefEA,x)
    return new_x

def extract_psd_features(X, sfreq, bands):
    """
    提取多频段 PSD 特征。

    参数：
    - X: EEG 数据，shape=(N, C, T)
    - sfreq: 采样率 (Hz)
    - bands: dict，如 {'delta': (0.5, 4), 'theta': (4, 8), ...}

    返回：
    - features: shape=(N, C × B)，其中 B 为频段个数
    """
    N, C, T = X.shape
    B = len(bands)
    features = np.zeros((N, C * B))

    freqs, _ = welch(X[0, 0, :], fs=sfreq, nperseg=sfreq)

    for n in range(N):
        band_powers = []
        for c in range(C):
            f, psd = welch(X[n, c, :], fs=sfreq, nperseg=sfreq)
            for band in bands.values():
                fmin, fmax = band
                idx_band = np.logical_and(f >= fmin, f <= fmax)
                # 积分或均值作为特征值
                power = np.mean(psd[idx_band])  # 或 np.trapz(psd[idx_band], f[idx_band])
                band_powers.append(power)
        features[n] = np.array(band_powers)
    return features

def extract_band_power_features(X, sfreq, bands):
    # # 提取频带功率
    # N, C, T = X.shape
    # band_power = []
    # for band_name, (fmin, fmax) in bands.items():
    #     X_band = np.zeros_like(X)
    #     for c in range(C):
    #         X_band[:, c, :] = filter_data(X[:, c, :], sfreq, l_freq=fmin, h_freq=fmax, verbose='error')
    #     # X_band = EA(X_band) # 加EA
    #     power = np.mean(X_band ** 2, axis=2)  # (N, C)
    #     band_power.append(power)
    # features = np.concatenate(band_power, axis=1)  # shape: (N, C×B)
    # return features

    """
    对每个 trial 提取基本特征，返回 shape (N, C×6)
    """
    feat_list = []
    for trial in X:  # trial: shape (C, T)
        feats = extract_basic_features(trial)  # shape: (C×6,)
        feat_list.append(feats)
    features = np.stack(feat_list, axis=0)  # shape: (N, C×6)
    return features


def extract_basic_features(signal):
    """
    对每个通道提取 6 个统计特征，最后拼接为 (C×6,)
    signal: shape (C, T)
    """
    feature_list = []
    C, T = signal.shape
    for c in range(C):
        ch = signal[c]  # 取出第 c 个通道 (T,)
        ch = (ch - np.mean(ch)) / np.std(ch)

        mean = np.mean(ch)
        std = np.std(ch)
        sample_entropy = np.log(np.std(np.diff(ch)))
        fuzzy_entropy = -np.log(euclidean(ch[:-1], ch[1:]) / len(ch))
        skewness_val = skew(ch)
        kurt = kurtosis(ch)

        feature_list.extend([mean, std, sample_entropy, fuzzy_entropy, skewness_val, kurt])

    return np.array(feature_list)  # shape: (C×6,)

def get_trial_slice(trial_counts, subject_id):
    if not 1 <= subject_id <= len(trial_counts):
        raise ValueError(f"subject_id must be between 1 and {len(trial_counts)}")

    start = sum(trial_counts[:subject_id - 1])
    end = start + trial_counts[subject_id - 1]
    return start, end

def get_ep_data(dataset_name, sub, para, duration=30.0, l_freq=1., h_freq=40.):
    a1 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/00_epilepsy/aaaaaanr/s001_2003/02_tcp_le/aaaaaanr_s001_t001.edf'
    a2 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/00_epilepsy/aaaaaawu/s001_2003/02_tcp_le/aaaaaawu_s001_t001.edf'
    a3 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/00_epilepsy/aaaaabdn/s001_2003/02_tcp_le/aaaaabdn_s001_t000.edf'
    a4 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/00_epilepsy/aaaaabhz/s001_2003/02_tcp_le/aaaaabhz_s001_t001.edf'
    a5 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/00_epilepsy/aaaaabju/s001_2003/02_tcp_le/aaaaabju_s001_t000.edf'

    n1 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/01_no_epilepsy/aaaaaebo/s001_2006/02_tcp_le/aaaaaebo_s001_t000.edf'
    n2 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/01_no_epilepsy/aaaaafiy/s001_2006/02_tcp_le/aaaaafiy_s001_t000.edf'
    n3 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/01_no_epilepsy/aaaaaigj/s001_2009/02_tcp_le/aaaaaigj_s001_t000.edf'
    n4= '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/01_no_epilepsy/aaaaajgn/s001_2009/02_tcp_le/aaaaajgn_s001_t000.edf'
    n5 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/01_no_epilepsy/aaaaakim/s002_2010/02_tcp_le/aaaaakim_s002_t001.edf'
    abnormal_list = [a1, a2, a3, a4, a5]
    normal_list = [n1, n2, n3, n4, n5]
    # eeg_chans = [
    #     'EEG A1-LE', 'EEG A2-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG CZ-LE', 'EEG F3-LE',
    #     'EEG F4-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG FP1-LE', 'EEG FP2-LE', 'EEG FZ-LE',
    #     'EEG O1-LE', 'EEG O2-LE', 'EEG OZ-LE', 'EEG P3-LE', 'EEG P4-LE', 'EEG PG1-LE',
    #     'EEG PG2-LE', 'EEG PZ-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE',
    #     'EEG 28-LE', 'EEG 29-LE', 'EEG 30-LE'
    # ]
    # eeg_chans = [
    #     'EEG FP1-LE', 'EEG FP2-LE',
    #     'EEG F3-LE', 'EEG F4-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG FZ-LE',
    #     'EEG C3-LE', 'EEG C4-LE', 'EEG CZ-LE',
    #     'EEG P3-LE', 'EEG P4-LE', 'EEG PZ-LE',
    #     'EEG O1-LE', 'EEG O2-LE', 'EEG OZ-LE',
    #     'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE',
    #     'EEG A1-LE', 'EEG A2-LE'  # 可作为参考通道
    # ]
    eeg_chans = [
        'EEG FP1-LE',  # 左额极（额叶前部）
        'EEG FP2-LE',  # 右额极
        'EEG F7-LE',   # 左额颞部
        'EEG F8-LE',   # 右额颞部
        'EEG C3-LE',   # 中央区左侧
        'EEG C4-LE',   # 中央区右侧
        'EEG O1-LE',   # 左枕叶
        'EEG O2-LE'    # 右枕叶
    ]
    X_list, y_list = [], []
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30)
    }

    count_list = []
    # 处理 normal 文件（label = 0）
    for file_path in normal_list:
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
            raw.pick(eeg_chans)
            raw.filter(l_freq, h_freq, verbose='error')
            epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True, verbose='error')
            X = epochs.get_data()
            y = np.array(["normal"] * len(X))
            sfreq = raw.info["sfreq"]
            X_feat = extract_band_power_features(X, sfreq, bands)
            # X_feat = extract_psd_features(X, sfreq, bands)
            X_list.append(X_feat)
            y_list.append(y)
            count_list.append(X_feat.shape[0])
        except Exception as e:
            print(f"[NOR] 读取失败: {file_path} | 错误: {e}")

    # 处理 abnormal 文件（label = 1）
    for file_path in abnormal_list:
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
            raw.pick(eeg_chans)
            raw.filter(l_freq, h_freq, verbose='error')
            epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True, verbose='error')
            X = epochs.get_data()
            y = np.array(["abnormal"] * len(X))
            sfreq = raw.info["sfreq"]
            X_feat = extract_band_power_features(X, sfreq, bands)
            # X_feat = extract_psd_features(X, sfreq, bands)
            X_list.append(X_feat)
            y_list.append(y)
            count_list.append(X_feat.shape[0])
        except Exception as e:
            print(f"[ABN] 读取失败: {file_path} | 错误: {e}")

    # 拼接全部样本
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    y_all = y_all.astype(str)

    print(f"✅ 总共样本数: {X_all.shape[0]}, 特征维度: {X_all[0].shape}, 每个被试的样本数: count_list")

    start, end = get_trial_slice(count_list, sub)
    train_data = np.concatenate([X_all[:start], X_all[end:]], axis=0)
    test_data = X_all[start:end]
    train_labels = np.concatenate([y_all[:start], y_all[end:]], axis=0)
    test_labels = y_all[start:end]

    return train_data, test_data, train_labels, test_labels

# def get_ep_data_cv(dataset_name, sub_index, para='epilepsy', k=5):
#     # 获取所有 EEG 特征和标签
#     a1 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/00_epilepsy/aaaaaanr/s001_2003/02_tcp_le/aaaaaanr_s001_t001.edf'
#     a2 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/00_epilepsy/aaaaaawu/s001_2003/02_tcp_le/aaaaaawu_s001_t001.edf'
#     a3 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/00_epilepsy/aaaaabdn/s001_2003/02_tcp_le/aaaaabdn_s001_t000.edf'
#     a4 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/00_epilepsy/aaaaabhz/s001_2003/02_tcp_le/aaaaabhz_s001_t001.edf'
#     a5 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/00_epilepsy/aaaaabju/s001_2003/02_tcp_le/aaaaabju_s001_t000.edf'

#     n1 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/01_no_epilepsy/aaaaaebo/s001_2006/02_tcp_le/aaaaaebo_s001_t000.edf'
#     n2 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/01_no_epilepsy/aaaaafiy/s001_2006/02_tcp_le/aaaaafiy_s001_t000.edf'
#     n3 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/01_no_epilepsy/aaaaaigj/s001_2009/02_tcp_le/aaaaaigj_s001_t000.edf'
#     n4= '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/01_no_epilepsy/aaaaajgn/s001_2009/02_tcp_le/aaaaajgn_s001_t000.edf'
#     n5 = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/01_no_epilepsy/aaaaakim/s002_2010/02_tcp_le/aaaaakim_s002_t001.edf'
#     abnormal_list = [a1, a2, a3, a4, a5]
#     normal_list = [n1, n2, n3, n4, n5]

#     X, labels = get_ep_data(abnormal_list, normal_list, duration=30.0)
#     labels = labels.astype(str)

#     # 创建 stratified k-fold
#     # skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
#     skf = StratifiedKFold(n_splits=k, shuffle=True)

#     for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = labels[train_idx], labels[test_idx]

#         # # 可选：归一化（fit on train, transform on both）
#         # scaler = StandardScaler()
#         # X_train = scaler.fit_transform(X_train)
#         # X_test = scaler.transform(X_test)

#         print(f"[Fold {fold_idx}] train: {X_train.shape}, test: {X_test.shape}")
#         yield X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # 需要解决通道数量不一致问题
    abnormal_file = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/00_epilepsy/aaaaaanr/s001_2003/02_tcp_le/aaaaaanr_s001_t001.edf'
    normal_file = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/01_no_epilepsy/aaaaaebo/s001_2006/02_tcp_le/aaaaaebo_s001_t000.edf'
    abnormal_list = [abnormal_file]
    normal_list = [normal_file]


    get_ep_data('TUEP', 1, 'epilepsy')