# import os
# import sys
# import torch
# current_file_path = os.path.abspath(__file__)
# parent_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
# sys.path.append(parent_parent_dir)

# import argparse
# from tqdm import tqdm
# import numpy as np
# from sklearn.cross_decomposition import CCA
# from sklearn.metrics import accuracy_score
# from moabb.datasets import BNCI2014001, BNCI2015_004, Weibo2014, Nakanishi2015, BNCI2014008
# from moabb.paradigms import MotorImagery, SSVEP
# from PGAP.load.load_five_folds import get_moabb_data_cv

# parser = argparse.ArgumentParser()
# parser.add_argument('--sub_index', type=int, default=1, help='1-9')
# parser.add_argument('--dataset_name', type=str, default='BNCI2014008')
# args = parser.parse_args()

# para_dict = {
#     '2a': 'mi',
#     '2b': 'mi',
#     'BNCI2015_001': 'mi',
#     'BNCI2014_002': 'mi',
#     'Kalunga2016': 'ssvep',
#     'Nakanishi2015': 'ssvep',
#     'BNCI2014008': 'mi',
# }
# dataset_name = args.dataset_name
# sub_index = args.sub_index

# def ndToList(array):
#     return [array[i] for i in range(array.shape[0])]

# def get_Reference_Signal(num_harmonics, targets, sf, timepts):
#     reference_signals = []
#     t = np.arange(0, (timepts / sf), step=1.0 / sf)
#     for f in targets:
#         reference_f = []
#         for h in range(1, num_harmonics + 1):
#             reference_f.append(np.sin(2 * np.pi * h * f * t)[0:timepts])
#             reference_f.append(np.cos(2 * np.pi * h * f * t)[0:timepts])
#         reference_signals.append(reference_f)
#     reference_signals = np.asarray(reference_signals)
#     return reference_signals

# def find_correlation(n_components, X, Y):
#     cca = CCA(n_components)
#     corr = np.zeros(n_components)
#     num_freq = Y.shape[0]
#     result = np.zeros(num_freq)
#     for freq_idx in range(0, num_freq):
#         matched_X = X

#         cca.fit(matched_X.T, Y[freq_idx].T)
#         # cca.fit(X.T, Y[freq_idx].T)
#         x_a, y_b = cca.transform(matched_X.T, Y[freq_idx].T)
#         for i in range(0, n_components):
#             corr[i] = np.corrcoef(x_a[:, i], y_b[:, i])[0, 1]
#             result[freq_idx] = np.max(corr)
#     return result

# for fold_idx, (train_X, test_X, train_y, test_y) in enumerate(
#     tqdm(get_moabb_data_cv(dataset_name, sub_index, para_dict[dataset_name]), total=5, desc=f"CV for {dataset_name}-sub{sub_index}")
# ):
#     train_data = ndToList(train_X)
#     test_data = ndToList(test_X)
#     train_labels = train_y.tolist()
#     test_labels = test_y.tolist()

#     test_data = train_data
#     test_labels = train_labels

#     sf = 256
#     timepts = test_data[0].shape[-1]

#     targets = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
#     reference_signals = get_Reference_Signal(3, targets, sf, timepts)

#     predicts = []
#     for i in range(len(test_data)):
#         result = find_correlation(1, test_data[i], reference_signals)
#         predicts.append(str(targets[np.argmax(result)]))

#     accuracy = accuracy_score(test_labels, predicts)
#     # print(predicts, test_labels)
#     print(accuracy)


from moabb.datasets import BNCI2014_002, BNCI2015_001
from moabb.paradigms import LeftRightImagery, MotorImagery

dataset = BNCI2015_001()
paradigm = MotorImagery()

def get_first_index(dataframe, column_index, target_value):
    """
    获取 DataFrame 的第 column_index 列中，第一个等于 target_value 的索引。

    参数:
    - dataframe: pandas.DataFrame，数据表
    - column_index: int，要查询的列索引（从 0 开始）
    - target_value: str，要查找的值

    返回:
    - 第一个匹配项的索引，如果不存在返回 None
    """
    # 获取第 column_index 列的数据
    column_data = dataframe.iloc[:, column_index]

    # 查找第一个匹配项的索引
    result = column_data[column_data == target_value].index
    return result[0] if not result.empty else None

X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[12])
print(X.shape, y, metadata)
index = get_first_index(metadata, column_index=1, target_value="1B") # number of training trials
print(index)


# import numpy as np
# import pandas as pd
# sub_index=1
# all_chans =  [
#     "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6",
#     "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5",
#     "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2",
#     "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7",
#     "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2", "VEO", "HEO",
# ]

# mi_related_channels = ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
#  'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P3', 'P1', 'P2', 'P4']

# def get_channel_indices(mi_channels, all_channels):
#     return [all_channels.index(ch) for ch in mi_channels]

# indices = get_channel_indices(mi_related_channels, all_chans)
# chans = np.array(indices)

# X = np.load('data/Weibo2014/X.npy')
# X = X[:, chans, :]
# labels = np.load('data/Weibo2014/labels.npy')
# meta = pd.read_csv('data/Weibo2014/meta.csv')
# start = 560*(sub_index-1)
# X = X[start: start+560]
# labels = labels[start: start+560]
# meta = meta[start: start+560]

# mask = (labels == 'left_hand') | (labels == 'right_hand')
# X = X[mask]
# labels = labels[mask]
# meta = meta[mask]


# print(X.shape)
# sub_index = 10
# def get_start_and_length(sub_index):
#     lengths = [560] * 10
#     lengths[5] = 500  # 第 6 个（下标 5）长度为 500

#     start = sum(lengths[:sub_index - 1])  # 所有前面段的长度之和
#     length = lengths[sub_index - 1]
#     return start, length
# start, length = get_start_and_length(sub_index)
# print(start, length)

# import os
# import numpy as np
# import mne
# from mne.datasets.sleep_physionet.age import fetch_data
# from mne.filter import filter_data
# from sklearn.model_selection import StratifiedKFold

# def extract_band_power_features(X, sfreq, bands):
#     N, C, T = X.shape
#     band_power = []

#     for band_name, (fmin, fmax) in bands.items():
#         X_band = np.zeros_like(X)
#         for c in range(C):
#             X_band[:, c, :] = filter_data(X[:, c, :], sfreq, l_freq=fmin, h_freq=fmax, verbose='error')
#         power = np.mean(X_band ** 2, axis=2)  # (N, C)
#         band_power.append(power)

#     features = np.concatenate(band_power, axis=1)  # shape: (N, C×B)
#     return features


# def get_ep_data(abnormal_list, normal_list, duration=5.0, l_freq=1., h_freq=40.):
#     # eeg_chans = [
#     #     'EEG A1-LE', 'EEG A2-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG CZ-LE', 'EEG F3-LE',
#     #     'EEG F4-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG FP1-LE', 'EEG FP2-LE', 'EEG FZ-LE',
#     #     'EEG O1-LE', 'EEG O2-LE', 'EEG OZ-LE', 'EEG P3-LE', 'EEG P4-LE', 'EEG PG1-LE',
#     #     'EEG PG2-LE', 'EEG PZ-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE',
#     #     'EEG 28-LE', 'EEG 29-LE', 'EEG 30-LE'
#     # ]
#     # eeg_chans = [
#     #     'EEG FP1-LE', 'EEG FP2-LE',
#     #     'EEG F3-LE', 'EEG F4-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG FZ-LE',
#     #     'EEG C3-LE', 'EEG C4-LE', 'EEG CZ-LE',
#     #     'EEG P3-LE', 'EEG P4-LE', 'EEG PZ-LE',
#     #     'EEG O1-LE', 'EEG O2-LE', 'EEG OZ-LE',
#     #     'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG T6-LE',
#     #     'EEG A1-LE', 'EEG A2-LE'  # 可作为参考通道
#     # ]
#     eeg_chans = [
#         'EEG FP1-LE',  # 左额极（额叶前部）
#         'EEG FP2-LE',  # 右额极
#         'EEG F7-LE',   # 左额颞部
#         'EEG F8-LE',   # 右额颞部
#         'EEG C3-LE',   # 中央区左侧
#         'EEG C4-LE',   # 中央区右侧
#         'EEG O1-LE',   # 左枕叶
#         'EEG O2-LE'    # 右枕叶
#     ]
#     X_list, y_list = [], []
#     bands = {
#         "delta": (0.5, 4),
#         "theta": (4, 8),
#         "alpha": (8, 12),
#         "sigma": (12, 16),
#     }

#     # 处理 abnormal 文件（label = 1）
#     for file_path in abnormal_list:
#         try:
#             raw = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
#             raw.pick(eeg_chans)
#             raw.filter(l_freq, h_freq, verbose='error')
#             epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True, verbose='error')
#             X = epochs.get_data()
#             y = np.array(["abnormal"] * len(X))
#             sfreq = raw.info["sfreq"]
#             X_feat = extract_band_power_features(X, sfreq, bands)
#             X_list.append(X_feat)
#             y_list.append(y)
#         except Exception as e:
#             print(f"[ABN] 读取失败: {file_path} | 错误: {e}")

#     # 处理 normal 文件（label = 0）
#     for file_path in normal_list:
#         try:
#             raw = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
#             raw.pick(eeg_chans)
#             raw.filter(l_freq, h_freq, verbose='error')
#             epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True, verbose='error')
#             X = epochs.get_data()
#             y = np.array(["normal"] * len(X))
#             sfreq = raw.info["sfreq"]
#             X_feat = extract_band_power_features(X, sfreq, bands)
#             X_list.append(X_feat)
#             y_list.append(y)
#         except Exception as e:
#             print(f"[NOR] 读取失败: {file_path} | 错误: {e}")

#     # 拼接全部样本
#     X_all = np.concatenate(X_list, axis=0)
#     y_all = np.concatenate(y_list, axis=0)

#     print(f"✅ 总共样本数: {X_all.shape[0]}, 特征维度: {X_all.shape[1:]}")

#     return X_all, y_all

# def get_ep_data_cv(dataset_name, sub_index, para='epilepsy', k=5):
#     # 获取所有 EEG 特征和标签
#     abnormal_file = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/00_epilepsy/aaaaaanr/s001_2003/02_tcp_le/aaaaaanr_s001_t001.edf'
#     normal_file = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/01_no_epilepsy/aaaaaebo/s001_2006/02_tcp_le/aaaaaebo_s001_t000.edf'
#     abnormal_list = [abnormal_file]
#     normal_list = [normal_file]

#     X, labels = get_ep_data(abnormal_list, normal_list, duration=5.0)
#     labels = labels.astype(str)

#     # 创建 stratified k-fold
#     skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

#     for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
#         X_train, X_test = X[train_idx], X[test_idx]
#         y_train, y_test = labels[train_idx], labels[test_idx]

#         # # 可选：归一化（fit on train, transform on both）
#         # scaler = StandardScaler()
#         # X_train = scaler.fit_transform(X_train)
#         # X_test = scaler.transform(X_test)

#         print(f"[Fold {fold_idx}] train: {X_train.shape}, test: {X_test.shape}")
#         yield X_train, X_test, y_train, y_test

# if __name__ == '__main__':
#     # 需要解决通道数量不一致问题
#     abnormal_file = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/00_epilepsy/aaaaaanr/s001_2003/02_tcp_le/aaaaaanr_s001_t001.edf'
#     normal_file = '/data1/labram_data/tuh_eeg/tuh_eeg_epilepsy/v2.0.1/01_no_epilepsy/aaaaaebo/s001_2006/02_tcp_le/aaaaaebo_s001_t000.edf'
#     abnormal_list = [abnormal_file]
#     normal_list = [normal_file]

#     get_ep_data_cv('tuep', 1)