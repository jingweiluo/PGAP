import os
import numpy as np
import mne
from mne.datasets.sleep_physionet.age import fetch_data
from mne.filter import filter_data
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import euclidean
from scipy.stats import skew, kurtosis
from scipy.signal import welch

# root_dir = "/data1/labram_data/"
root_dir = ""
data_dir = "physionet-sleep-data/"

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
    N, C, T = X.shape
    band_power = []

    for band_name, (fmin, fmax) in bands.items():
        X_band = np.zeros_like(X)
        for c in range(C):
            X_band[:, c, :] = filter_data(X[:, c, :], sfreq, l_freq=fmin, h_freq=fmax, verbose='error')
        power = np.mean(X_band ** 2, axis=2)  # (N, C)
        band_power.append(power)

    features = np.concatenate(band_power, axis=1)  # shape: (N, C×B)
    return features

# def extract_band_power_features(X, sfreq, bands):
#     """
#     对每个 trial 提取基本特征，返回 shape (N, C×6)
#     """
#     feat_list = []
#     for trial in X:  # trial: shape (C, T)
#         feats = extract_basic_features(trial)  # shape: (C×6,)
#         feat_list.append(feats)
#     features = np.stack(feat_list, axis=0)  # shape: (N, C×6)
#     return features

# def extract_basic_features(signal):
#     """
#     对每个通道提取 6 个统计特征，最后拼接为 (C×6,)
#     signal: shape (C, T)
#     """
#     feature_list = []
#     C, T = signal.shape
#     for c in range(C):
#         ch = signal[c]  # 取出第 c 个通道 (T,)
#         ch = (ch - np.mean(ch)) / np.std(ch)

#         mean = np.mean(ch)
#         std = np.std(ch)
#         sample_entropy = np.log(np.std(np.diff(ch)))
#         fuzzy_entropy = -np.log(euclidean(ch[:-1], ch[1:]) / len(ch))
#         skewness_val = skew(ch)
#         kurt = kurtosis(ch)

#         feature_list.extend([mean, std, sample_entropy, fuzzy_entropy, skewness_val, kurt])

#     return np.array(feature_list)  # shape: (C×6,)


def get_sleep_data(sub_indices, chans=['EEG Fpz-Cz', 'EEG Pz-Oz']):
    eeg_channel = chans
    # bands = {
    #     "delta": (0.5, 4),
    #     "theta": (4, 8),
    #     "alpha": (8, 12),
    #     "sigma": (12, 16),
    # }
    bands = {
        'delta':   (0.5, 4),
        'theta':   (4, 8),
        'alpha':   (8, 13),
        'spindle': (12, 14),
        'beta':    (14, 30)
    }

    all_features = []
    all_labels = []
    all_counts = [] # 用来记录每个被试的trial数量

    for subject in sub_indices:
        psg_file = os.path.join(root_dir, data_dir, f'SC4{subject:02d}1E0-PSG.edf')
        hypnogram_file = os.path.join(root_dir, data_dir, f'SC4{subject:02d}1EC-Hypnogram.edf')

        # 若本地不存在则尝试下载
        if not os.path.exists(psg_file) or not os.path.exists(hypnogram_file):
            print(f"尝试下载受试者 {subject} 的数据...")
            try:
                fetch_data(subjects=[subject], recording=[1], path=root_dir, on_missing='raise')
            except Exception as e:
                print(f"受试者 {subject} 数据下载失败: {e}")
                continue

        # 再次检查文件是否存在
        if not os.path.exists(psg_file) or not os.path.exists(hypnogram_file):
            print(f"跳过受试者 {subject}，文件仍然不存在")
            continue

        try:
            raw = mne.io.read_raw_edf(psg_file, preload=True, verbose='error')
            annotations = mne.read_annotations(hypnogram_file)
            raw.set_annotations(annotations)

            # 裁剪多余段
            annotations.crop(annotations[1]["onset"] - 30 * 60, annotations[-2]["onset"] + 30 * 60)

            annotation_desc_2_event_id = {
                "Sleep stage W": 1,
                "Sleep stage 1": 2,
                "Sleep stage 2": 3,
                "Sleep stage 3": 4,
                "Sleep stage 4": 4,
                "Sleep stage R": 5,
            }

            raw.pick(eeg_channel)
            raw.set_annotations(annotations)

            events, _ = mne.events_from_annotations(
                raw, event_id=annotation_desc_2_event_id, chunk_duration=30.0
            )

            event_id = {
                "Sleep stage W": 1,
                "Sleep stage 1": 2,
                "Sleep stage 2": 3,
                "Sleep stage 3/4": 4,
                "Sleep stage R": 5,
            }

            tmax = 30.0 - 1.0 / raw.info["sfreq"]
            epochs = mne.Epochs(
                raw=raw,
                events=events,
                event_id=event_id,
                tmin=0.0,
                tmax=tmax,
                baseline=None,
                preload=True,
                verbose="error"
            )

            if len(epochs) == 0:
                print(f"受试者 {subject} 没有有效 epoch")
                continue

            X = epochs.get_data()
            y = epochs.events[:, -1]
            sfreq = raw.info["sfreq"]
            # X_feat = extract_band_power_features(X, sfreq, bands)
            X_feat = extract_psd_features(X, sfreq, bands)

            all_features.append(X_feat)
            all_labels.append(y)
            all_counts.append(len(y))

        except Exception as e:
            print(f"受试者 {subject} 处理失败: {e}")
            continue

    if not all_features:
        raise ValueError("没有可用的数据，请检查数据路径或被试编号")

    X_all = np.concatenate(all_features, axis=0)
    y_all = np.concatenate(all_labels, axis=0)
    y_all = y_all.astype(str)

    return X_all, y_all, all_counts

def get_sleep_data_cv(dataset_name, sub_index, para='sleep', k=5, chans=['EEG Fpz-Cz', 'EEG Pz-Oz']):
    # 获取所有 EEG 特征和标签
    X, labels, _ = get_sleep_data(sub_indices=[sub_index], chans=chans)
    labels = labels.astype(str)

    # 创建 stratified k-fold
    skf = StratifiedKFold(n_splits=k, shuffle=True)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # # 可选：归一化（fit on train, transform on both）
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        print(f"[Fold {fold_idx}] train: {X_train.shape}, test: {X_test.shape}")
        yield X_train, X_test, y_train, y_test

def get_sleep_data_crs_sub(dataset_name, sub_index, para='sleep',chans=['EEG Fpz-Cz', 'EEG Pz-Oz']):
    if dataset_name == 'sleep-edfx':
        total_subs = 7

    # 获取所有 EEG 特征和标签
    X, labels, all_counts = get_sleep_data(sub_indices=list(range(1, total_subs+1)), chans=chans)
    labels = labels.astype(str)

    # 根据测试被试index，计算
    test_start = sum(all_counts[:sub_index-1])
    test_length = all_counts[sub_index-1]

    X_train = np.r_[X[:test_start], X[test_start+test_length:]]
    y_train = np.r_[labels[:test_start], labels[test_start+test_length:]]
    X_test = X[test_start:test_start+test_length]
    y_test = labels[test_start:test_start+test_length]


    print(f"train: {X_train.shape}, {y_train.shape}, test: {X_test.shape}, {y_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    get_sleep_data_crs_sub('sleep-edfx', 7)
