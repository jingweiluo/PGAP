"""
================================
Tutorial 1: Simple Motor Imagery
================================

In this example, we will go through all the steps to make a simple BCI
classification task, downloading a dataset and using a standard classifier. We
choose the dataset 2a from BCI Competition IV, a motor imagery task. We will
use a CSP to enhance the signal-to-noise ratio of the EEG epochs and a LDA to
classify these signals.
"""

# Authors: Pedro L. C. Rodrigues, Sylvain Chevallier
#
# https://github.com/plcrodrigues/Workshop-MOABB-BCI-Graz-2019

import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mne.decoding import CSP
from mne.preprocessing import Xdawn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
import scipy.signal as signal
from sklearn.cross_decomposition import CCA
import mne
import moabb
from moabb.datasets import BNCI2014_001, BNCI2014_002, BNCI2014_004, BNCI2015_001, Kalunga2016, Nakanishi2015, BNCI2014008, Weibo2014
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery, MotorImagery, SSVEP, P300

import numpy as np

import os
import sys
import torch
current_file_path = os.path.abspath(__file__)
parent_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(parent_parent_dir)
# from PGAP.model.trainer import EEGNet_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


moabb.set_log_level("info")
warnings.filterwarnings("ignore")

from scipy.signal import butter, filtfilt
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    对信号进行频带滤波。

    参数:
    - data: 输入的信号 (1D 或 2D 数组)。
    - lowcut: 低频截止频率 (Hz)。
    - highcut: 高频截止频率 (Hz)。
    - fs: 采样率 (Hz)。
    - order: 滤波器阶数 (默认为 4)。

    返回:
    - 滤波后的信号 (与输入 shape 相同)。
    """
    nyquist = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyquist
    high = highcut / nyquist

    # 设计滤波器
    b, a = butter(order, [low, high], btype='band')

    # 应用滤波器
    filtered_data = filtfilt(b, a, data, axis=-1)  # 支持多维数据，沿最后一维滤波
    return filtered_data

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

##############################################################################
# Instantiating Dataset
# ---------------------
#
# The first thing to do is to instantiate the dataset that we want to analyze.
# MOABB has a list of many different datasets, each one containing all the
# necessary information for describing them, such as the number of subjects,
# size of trials, names of classes, etc.
#
# The dataset class has methods for:
#
# - downloading its files from some online source (e.g. Zenodo)
# - importing the data from the files in whatever extension they might be
#   (like .mat, .gdf, etc.) and instantiate a Raw object from the MNE package

dataset_dict = {
    '2a': BNCI2014_001(),
    '2b': BNCI2014_004(),
    'BNCI2015_001': BNCI2015_001(),
    'BNCI2014_002': BNCI2014_002(),
    'Kalunga2016': Kalunga2016(),
    'Nakanishi2015': Nakanishi2015(),
    'BNCI2014008': BNCI2014008(),
    'Weibo2014': Weibo2014(),
}
paradigm_dict = {
    '2a': LeftRightImagery(),
    '2b': LeftRightImagery(),
    'BNCI2015_001': MotorImagery(),
    'BNCI2014_002': MotorImagery(),
    'Kalunga2016': SSVEP(n_classes=4),
    'Nakanishi2015': SSVEP(n_classes=12),
    'BNCI2014008': P300(),
    'Weibo2014': MotorImagery(),
}

sfreq_dict = {
    '2a': 250,
    '2b': 250,
    'BNCI2015_001': 512,
    'BNCI2014_002': 512,
    'Kalunga2016': 256,
    'Nakanishi2015': 256,
    'BNCI2014008': 256,
    'Weibo2014': 200,
}

# ratio_dict = { #关系到使用EEGNet提取特征用到的训练集和验证集的大小
#     '2a': [0.75, 0.25, 1],
#     # '2a': [0.875, 0.125, 1],
#     '2b': [],
#     'BNCI2015_001': [0.8, 0.2, 1],
#     'BNCI2014_002': [80, 20, 60],
#     'Kalunga2016': [0.75, 0.25, 1],
#     'Nakanishi2015': [0.7, 0.1, 0.2],
# }

num_chans_dict = {
    '2a': 22,
    '2b': 3,
    'BNCI2015_001': 13,
    'BNCI2014_002': 15,
    'Kalunga2016': 8,
    'Nakanishi2015': 8,
    'BNCI2014008': 8,
    'Weibo2014': 60, # 21
}

num_cls_dict = {
    '2a': 2,
    '2b': 2,
    'BNCI2015_001': 2,
    'BNCI2014_002': 2,
    'Kalunga2016': 4,
    'Nakanishi2015': 12,
    'BNCI2014008': 2,
    'Weibo2014': 2,
}

ssvep_stim_freq_dict = {
    'Kalunga2016': [13.0, 17.0, 21.0],
    'Nakanishi2015': [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75],
}
# train_ratio_dict = { #关系到使用CSP或者CCA等手工提取特征用到的训练集大小
#     '2a': 0.5,
#     'BNCI2015_001': 0.5,
#     'BNCI2014_002': 0.625,
#     'Kalunga2016': 0.5,
#     'Nakanishi2015': 0.8,
# }
# num_train_dict = {
#     '2a': 144,
#     'BNCI2015_001': 200,
#     'BNCI2014_002': 100,
#     'Kalunga2016': 32,
#     'Nakanishi2015': 90, # 一共180trials
# }
def select_n_per_class(data, label, n, random_state=42):
    """
    对数量太大的数据集，从每个类别中随机选择 n 个样本

    参数:
    - data: ndarray, shape (N, C, T)
    - label: ndarray, shape (N,)
    - n: int, 每个类别选择的样本数
    - random_state: int or None, 用于控制随机性

    返回:
    - selected_data: ndarray, shape (n * num_classes, C, T)
    - selected_label: ndarray, shape (n * num_classes,)
    """
    np.random.seed(random_state)
    selected_indices = []

    unique_classes = np.unique(label)
    for cls in unique_classes:
        cls_indices = np.where(label == cls)[0]
        if len(cls_indices) < n:
            raise ValueError(f"类别 {cls} 的样本数不足：只有 {len(cls_indices)} 个，无法取 {n} 个")
        chosen = np.random.choice(cls_indices, n, replace=False)
        selected_indices.extend(chosen)

    selected_indices = np.array(selected_indices)
    return data[selected_indices], label[selected_indices]

from sklearn.model_selection import StratifiedKFold

def get_moabb_data_cv(dataset_name, sub_index, para, n_splits=5):
    """
    5折交叉验证，支持 MI + CSP 和 SSVEP + TRCA

    参数:
    - dataset_name: 数据集名称 ~["2a", "2b", ...]
    - sub_index: 被试编号，从1开始
    - para: "mi" 或 "ssvep"
    返回:
    - 一个 generator，每次迭代返回 train_data, test_data, train_labels, test_labels
    """

    # 获取元信息
    dataset = dataset_dict[dataset_name]
    paradigm = paradigm_dict[dataset_name]
    sfreq = sfreq_dict[dataset_name]
    num_cls = num_cls_dict[dataset_name]
    num_chans = num_chans_dict[dataset_name]

    if dataset_name == 'Weibo2014':
        all_chans =  [
            "Fp1", "Fpz", "Fp2", "AF3", "AF4", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6",
            "F8", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "T7", "C5",
            "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2",
            "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7",
            "PO5", "PO3", "POz", "PO4", "PO6", "PO8", "CB1", "O1", "Oz", "O2", "CB2", "VEO", "HEO",
        ]
        mi_related_channels = ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P3', 'P1', 'P2', 'P4']

        # mi_related_channels = [
        #     'FC3', 'FCz', 'FC4',     # 前运动皮层（运动准备区）
        #     'C3',  'Cz',  'C4',      # 运动皮层（左右手 + 双脚中线）
        #     'CP3', 'CPz', 'CP4'      # 体感皮层（运动反馈通路）
        # ]

        def get_channel_indices(mi_channels, all_channels):
            all_upper = [ch.upper() for ch in all_channels]
            return [all_upper.index(ch.upper()) for ch in mi_channels]

        indices = get_channel_indices(mi_related_channels, all_chans)
        chans = np.array(indices)

        X = np.load('data/Weibo2014/X.npy')
        X = X[:, chans, :]
        labels = np.load('data/Weibo2014/labels.npy')
        meta = pd.read_csv('data/Weibo2014/meta.csv')

        def get_start_and_length(sub_index):
            lengths = [560] * 10
            lengths[5] = 500  # 第 6 个（下标 5）长度为 500

            start = sum(lengths[:sub_index - 1])  # 所有前面段的长度之和
            length = lengths[sub_index - 1]
            return start, length
        start, length = get_start_and_length(sub_index)

        X = X[start: start+length]
        labels = labels[start: start+length]
        meta = meta[start: start+length]

        mask = (labels == 'feet') | (labels == 'right_hand')
        X = X[mask]
        labels = labels[mask]
        meta = meta[mask]

    else:
        # 加载数据
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[sub_index])
        if dataset_name == 'BNCI2014008':
            X, labels = select_n_per_class(X, labels, 200)

    print(f"[INFO] 数据 shape: {X.shape}, 标签 shape: {labels.shape}")
    # count_half = np.sum(labels[:len(labels)//2] == 'feet')
    # count_all = np.sum(labels == 'feet')
    # print('meta信息', meta)
    # print('全部labels，前一半数据中标签为feet的数量，全部数据中标签为feet的数量，全部数据数量', labels, count_half, count_all, len(labels))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
        # 用于提取EEGNet特征
        # X = bandpass_filter(X, lowcut=1, highcut=40, fs=sfreq) # 滤波1-40
        # fs_new = 128
        # num_samples_new = int(X.shape[-1] * (fs_new / sfreq))
        # X = signal.resample(X, num_samples_new, axis=-1) # 重采样
        # feat_map = EEGNet_transform(X, labels, para, [0.7, 0.1, 0.2], num_cls, num_chans) # 此处的ratio_list只使用2a数据集
        # train_feat, test_feat, y_train, y_test = feat_map[train_idx], feat_map[test_idx], labels[train_idx], labels[test_idx]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        if para == 'mi' or para == 'ssvep':
            n_components = 2

            # 每个频段训练一个 CSP
            csp_list = []
            for lowcut in range(8, 29, 4):
                highcut = lowcut + 4
                X_train_filtered = bandpass_filter(X_train, lowcut, highcut, fs=sfreq)
                csp = CSP(n_components=n_components, norm_trace=False)
                csp.fit(X_train_filtered, y_train)
                csp_list.append((lowcut, highcut, csp))

            def extract_csp_band_features(X_data, csp_list):
                feat_map = []
                for lowcut, highcut, csp_model in csp_list:
                    X_filtered = bandpass_filter(X_data, lowcut, highcut, fs=sfreq)
                    X_csp = csp_model.transform(X_filtered)
                    X_csp_expanded = X_csp[:, np.newaxis, :]
                    feat_map.append(X_csp_expanded)
                return np.concatenate(feat_map, axis=1)

            train_feat = extract_csp_band_features(X_train, csp_list)
            test_feat = extract_csp_band_features(X_test, csp_list)

            # 只保留前4个频段（即8–24Hz）
            train_feat = train_feat[:, 0:4, :]
            test_feat = test_feat[:, 0:4, :]

        elif para == 'p300':
            def make_epochs(data, sfreq=256, ch_names=None):
                n_trials, n_channels, n_times = data.shape
                if ch_names is None:
                    ch_names = [f'ch_{i}' for i in range(n_channels)]
                info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
                return mne.EpochsArray(data, info)

            # 1. 包装为 EpochsArray
            epochs_train = make_epochs(X_train, sfreq=sfreq)
            epochs_test = make_epochs(X_test, sfreq=sfreq)

            # 2. 拟合 xDAWN（只在训练集上）
            xdawn = Xdawn(n_components=1)
            xdawn.fit(epochs_train, y=y_train)

            # 3. 提取特征
            train_feat = xdawn.transform(epochs_train)  # shape: (N, n_components, T)
            test_feat = xdawn.transform(epochs_test)

            train_feat = train_feat.mean(axis=2)
            test_feat = test_feat.mean(axis=2)

        # elif para == 'ssvep':
        #     def get_Reference_Signal(num_harmonics, targets, sf, timepts):
        #         reference_signals = []
        #         t = np.arange(0, (timepts / sf), step=1.0 / sf)
        #         for f in targets:
        #             reference_f = []
        #             for h in range(1, num_harmonics + 1):
        #                 reference_f.append(np.sin(2 * np.pi * h * f * t)[0:timepts])
        #                 reference_f.append(np.cos(2 * np.pi * h * f * t)[0:timepts])
        #             reference_signals.append(reference_f)
        #         reference_signals = np.asarray(reference_signals)
        #         return reference_signals

        #     def find_correlation(n_components, X, Y):
        #         cca = CCA(n_components)
        #         corr = np.zeros(n_components)
        #         num_freq = Y.shape[0]
        #         result = np.zeros(num_freq)
        #         for freq_idx in range(0, num_freq):
        #             matched_X = X
        #             cca.fit(matched_X.T, Y[freq_idx].T)
        #             # cca.fit(X.T, Y[freq_idx].T)
        #             x_a, y_b = cca.transform(matched_X.T, Y[freq_idx].T)
        #             for i in range(0, n_components):
        #                 corr[i] = np.corrcoef(x_a[:, i], y_b[:, i])[0, 1]
        #                 result[freq_idx] = np.max(corr)
        #         return result

        #     targets = ssvep_stim_freq_dict[dataset_name]
        #     sf = sfreq_dict[dataset_name]
        #     timepts = X.shape[-1]
        #     reference_signals = get_Reference_Signal(3, targets, sf, timepts)

        #     feat_all = []
        #     for i in range(len(X)):
        #         feat = find_correlation(1, X[i], reference_signals)
        #         feat_all.append(feat)

        #     feat_all = np.array(feat_all)
        #     train_feat, test_feat = feat_all[train_idx], feat_all[test_idx]

        else:
            raise ValueError(f"不支持的范式类型: {para}")

        # Swap train and test for ablation study
        # train_feat, test_feat, y_train, y_test = test_feat, train_feat, y_test, y_train

        print(f"[Fold {fold_idx}] train: {train_feat.shape}, test: {test_feat.shape}")
        yield train_feat, test_feat, y_train, y_test

if __name__ == '__main__':
    get_moabb_data_cv("Weibo2014", 1, "mi")