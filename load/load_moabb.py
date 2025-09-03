import warnings
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
import scipy.signal as signal
from scipy.linalg import fractional_matrix_power
from sklearn.cross_decomposition import CCA

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

ratio_dict = { #关系到使用EEGNet提取特征用到的训练集和验证集的大小
    '2a': [0.75, 0.25, 1],
    # '2a': [0.875, 0.125, 1],
    '2b': [],
    'BNCI2015_001': [0.8, 0.2, 1],
    'BNCI2014_002': [80, 20, 60],
    'Kalunga2016': [0.75, 0.25, 1],
    'Nakanishi2015': [0.7, 0.1, 0.2],
    'BNCI2014008': [0.7, 0.1, 0.2],
    'Weibo2014': [0.7, 0.1, 0.2],
}

num_chans_dict = {
    '2a': 22,
    '2b': 3,
    'BNCI2015_001': 13,
    'BNCI2014_002': 15,
    'Kalunga2016': 8,
    'Nakanishi2015': 8,
    'BNCI2014008': 8,
    'Weibo2014': 21, # 原始60通道
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

ssvep_freq_dict = {
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
num_train_dict = {
    '2a': 144,
    '2b': 400,
    'BNCI2015_001': 200,
    'BNCI2014_002': 100,
    'Kalunga2016': 32,
    'Nakanishi2015': 90, # 一共180trials
    'BNCI2014008': 200,
    'Weibo2014': 80,
}

num_subs_dict = {
    '2a': 9,
    '2b': 9,
    'BNCI2015_001': 9,
    'BNCI2014_002': 14,
    'Kalunga2016': 12,
    'Nakanishi2015': 9, # 一共180trials
    'BNCI2014008': 8,
    'Weibo2014': 10,
}

def get_moabb_data(dataset_name, sub_index, testid, para):
    """
    获取MI训练及测试数据
    参数:
    - dataframe(string): 数据集名称 ~["2a", "2b"]
    - sub_index(int): 被试编号 ~[1-9]
    - testid(int): 测试集索引 ~[0, 1, 2, 3, 4]
    返回:
    - train_data(list[np(6*3)]): 特征值
    - train_labels(list[string]): 标签
    """
    dataset = dataset_dict[dataset_name]
    paradigm = paradigm_dict[dataset_name]
    sfreq = sfreq_dict[dataset_name]
    # 下面三个用于EEGNet特征提取
    ratio_list = ratio_dict[dataset_name]
    num_chans = num_chans_dict[dataset_name]
    num_cls = num_cls_dict[dataset_name]

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
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[sub_index])

    if dataset_name == 'Weibo2014' and sub_index == 6:
        num_train = 70
    elif dataset_name == 'BNCI2015_001' and (sub_index in range(8, 12)):
        num_train = 400
    elif dataset_name == '2b':
        if sub_index in [1,2,6,7,9]:
            num_train = 400
        elif sub_index in [3,4,5]:
            num_train = 420
        elif sub_index in [8]:
            num_train = 440
    else:
        num_train = num_train_dict[dataset_name]

    print('数据集meta信息:', meta)
    print('sfreq:', sfreq)
    print('训练集trial数量:', num_train)
    #============================这一部分是提取EEGNet特征==========================================
    # if para == 'mi':
    #     X = bandpass_filter(X, lowcut=4, highcut=32, fs=sfreq) # 滤波4-32
    #     fs_new = 128
    #     num_samples_new = int(X.shape[-1] * (fs_new / sfreq))
    #     X = signal.resample(X, num_samples_new, axis=-1) # 重采样
    #     feat_map = EEGNet_transform(X, labels, para, ratio_list, num_cls, num_chans) # 此处的ratio_list只使用2a数据集
    # elif para == 'ssvep':
    #     if dataset_name == 'Kalunga2016':
    #         X = X[:64] # 确保所有被试的trials一致
    #         print(X.shape)

    #     X = bandpass_filter(X, lowcut=1, highcut=40, fs=sfreq) # 滤波1-40
    #     fs_new = 128
    #     num_samples_new = int(X.shape[-1] * (fs_new / sfreq))
    #     X = signal.resample(X, num_samples_new, axis=-1) # 重采样
    #     feat_map = EEGNet_transform(X, labels, para, ratio_list, num_cls, num_chans) # 此处的ratio_list只使用2a数据集

    #============================这一部分是提取CSP特征==========================================
    if para == 'mi':
        n_components = 2  # 选择要分解的主成分数量
        csp = CSP(n_components=n_components, norm_trace=False)

        '''
        1 对X进行多个频段的滤波
        2 每个频段中使用训练数据训练CSP，并应用到全部数据上
        3 将多个频段的特征组合起来
        '''
        def get_freq_spatial_feat(X, labels, csp, sfreq):
            feat_map = []
            for lowcut in range(8,29,4):
                X_filtered = bandpass_filter(X, lowcut=lowcut, highcut=lowcut+4, fs=sfreq)
                csp.fit(X_filtered[:num_train], labels[:num_train])
                X_csp = csp.transform(X_filtered)
                X_csp_expanded = X_csp[:, np.newaxis, :]
                feat_map.append(X_csp_expanded)
            return np.concatenate(feat_map, axis=1)

        feat_map = get_freq_spatial_feat(X, labels, csp, sfreq) # num_trials, num_freq_bands, n_components 由于n_components < n_chans, 所以=n_chans
        feat_map = feat_map[:, 0:4, :] # 只取前面4个频段，也就是8-24Hz，这是通过knn预实验比较得到的，这个频段貌似效果最好，不过其他也可尝试
        # index = get_first_index(meta, column_index=1, target_value="1test") # number of training trials

    # ==================== 公共部分：切分数据集 ========================================
    if dataset_name == "2a":
        # 2a数据集分成了2个session，在不同天采集，分别由144，144个trial
        s1 = slice(0, 144)
        s2 = slice(144, 288)

        testid_dict = {
            0: s1,
            1: s2,
        }
    elif dataset_name == "2b":
        if sub_index in [1,2,6,7,9]:
            s1 = slice(0, 400)
            s2 = slice(400, 720)
        elif sub_index in [3,4,5]:
            s1 = slice(0, 420)
            s2 = slice(420, 740)
        elif sub_index in [8]:
            s1 = slice(0, 440)
            s2 = slice(440, 760)

        testid_dict = {
            0: s1,
            1: s2,
        }
    elif dataset_name == "BNCI2015_001":
        if sub_index in [8,9,10,11]:
            s1 = slice(0, 400)
            s2 = slice(400, 600)
        else:
            s1 = slice(0, 200)
            s2 = slice(200, 400)

        testid_dict = {
            0: s1,
            1: s2,
        }
    elif dataset_name == "BNCI2014_002":
        # 2014_002数据集每个被试1个session,其中包括8个run,前5个run为训练集，后3个run为测试集
        s1 = slice(0, 100)
        s2 = slice(100, 160)

        testid_dict = {
            0: s1,
            1: s2,
        }
    elif dataset_name == "Kalunga2016":
        s1 = slice(0, 32)
        s2 = slice(32, 64)

        testid_dict = {
            0: s1,
            1: s2,
        }
    elif dataset_name == "Nakanishi2015":
        s1 = slice(0, 90)
        s2 = slice(90, 180)

        testid_dict = {
            0: s1,
            1: s2,
        }
    elif dataset_name == "Weibo2014":
        if sub_index == 6:
            s1 = slice(0, 70)
            s2 = slice(70, 140)
        else:
            s1 = slice(0, 80)
            s2 = slice(80, 160)

        testid_dict = {
            0: s1,
            1: s2,
        }
    else: raise ValueError("该数据集不存在")

    train_data, test_data, train_labels, test_labels = np.delete(feat_map, testid_dict[testid], axis=0), feat_map[testid_dict[testid]], np.delete(labels, testid_dict[testid], axis=0), labels[testid_dict[testid]]
    print('数据的shape为:', train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
    print('测试集真实标签', test_labels)
    return train_data, test_data, train_labels, test_labels

def EA(x):
    new_x = np.zeros_like(x) #(N,C,T)
    cov = np.zeros((x.shape[0],x.shape[1],x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov,0)
    sqrtRefEA = fractional_matrix_power(refEA,-0.5)
    new_x = np.matmul(sqrtRefEA,x)
    return new_x

def get_start_and_length(sub_index, meta):
    # 自动查找包含 "subject" 的列名（不区分大小写）
    subject_col = [col for col in meta.columns if 'subject' in col.lower()]
    if not subject_col:
        raise ValueError("找不到包含 'subject' 的列")
    subject_col = subject_col[0]  # 取第一个匹配的列名

    # 获取该列并统计每个 subject 的 trial 数量
    subjects = meta[subject_col]
    lengths_by_subject = subjects.value_counts().sort_index()

    # 转换为列表
    lengths = lengths_by_subject.tolist()
    print('每个被试trials数量:', lengths)

    # 计算 start 和 length
    start = sum(lengths[:sub_index - 1])
    length = lengths[sub_index - 1]
    return start, length

def getEA(X, dataset_name, meta):
    """
    X:滤波后全部被试数据
    meta: 数据集的元信息. 注意: 要与X的长度对应, 比如如果X是筛选后的，meta也要筛选后的
    """
    ea_list = []
    for i in range(1, num_subs_dict[dataset_name]+1):
        start, length = get_start_and_length(i, meta)
        sub_X = X[start:start + length]
        ea_X = EA(sub_X)
        ea_list.append(ea_X)
    all_ea = np.concatenate(ea_list, axis=0)
    return all_ea

def get_moabb_data_crs_sub(dataset_name, sub_index, para):
    """
    获取MI训练及测试数据
    参数:
    - dataframe(string): 数据集名称 ~["2a", "2b"]
    - sub_index(int): 被试编号 ~[1-9]
    - testid(int): 测试集索引 ~[0, 1, 2, 3, 4]
    返回:
    - train_data(list[np(6*3)]): 特征值
    - train_labels(list[string]): 标签
    """
    dataset = dataset_dict[dataset_name]
    paradigm = paradigm_dict[dataset_name]
    num_subs = num_subs_dict[dataset_name]
    sfreq = sfreq_dict[dataset_name]
    # 下面三项用于EEGNet的特征提取
    ratio_list = ratio_dict[dataset_name]
    num_chans = num_chans_dict[dataset_name]
    num_cls = num_cls_dict[dataset_name]

    #================================获取全部被试数据 X, labels, meta，按顺序排列==========================================
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

        mask = (labels == 'feet') | (labels == 'right_hand')
        X = X[mask]
        labels = labels[mask]
        meta = meta[mask]
    else:
        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=list(range(1, num_subs+1)))

    print('数据量:', X.shape)
    print('标签量:', labels.shape)
    print('meta:', meta.head())
    #================================获取测试集的范围==========================================

    test_start, test_length = get_start_and_length(sub_index, meta)
    print('test_start_index, test_length', test_start, test_length)

    #============================这一部分是提取EEGNet特征==========================================
    # if para == 'mi':
    #     X = bandpass_filter(X, lowcut=4, highcut=32, fs=sfreq) # 滤波4-32
    #     fs_new = 128
    #     num_samples_new = int(X.shape[-1] * (fs_new / sfreq))
    #     X = signal.resample(X, num_samples_new, axis=-1) # 重采样
    #     feat_map = EEGNet_transform(X, labels, para, ratio_list, num_cls, num_chans) # 此处的ratio_list只使用2a数据集
    # elif para == 'ssvep':
    #     if dataset_name == 'Kalunga2016':
    #         X = X[:64] # 确保所有被试的trials一致
    #         print(X.shape)

    #     X = bandpass_filter(X, lowcut=1, highcut=40, fs=sfreq) # 滤波1-40
    #     fs_new = 128
    #     num_samples_new = int(X.shape[-1] * (fs_new / sfreq))
    #     X = signal.resample(X, num_samples_new, axis=-1) # 重采样
    #     feat_map = EEGNet_transform(X, labels, para, ratio_list, num_cls, num_chans) # 此处的ratio_list只使用2a数据集

    #============================这一部分是提取CSP特征==========================================
    if para == 'mi':
        n_components = 2  # 选择要分解的主成分数量
        csp = CSP(n_components=n_components, norm_trace=False)

        def get_freq_spatial_feat(X, labels, csp, sfreq):
            feat_map = []
            for lowcut in range(8,29,4):
                X_filtered = bandpass_filter(X, lowcut=lowcut, highcut=lowcut+4, fs=sfreq)
                X_filtered = getEA(X_filtered, dataset_name, meta) # EA
                csp.fit(np.r_[X_filtered[:test_start], X_filtered[test_start+test_length:]], np.r_[labels[:test_start], labels[test_start+test_length:]])
                X_csp = csp.transform(X_filtered)
                X_csp_expanded = X_csp[:, np.newaxis, :]
                feat_map.append(X_csp_expanded)
            return np.concatenate(feat_map, axis=1)

        feat_map = get_freq_spatial_feat(X, labels, csp, sfreq) # num_trials, num_freq_bands, n_components 由于n_components < n_chans, 所以=n_chans
        feat_map = feat_map[:, 0:4, :] # 只取前面4个频段，也就是8-24Hz，这是通过knn预实验比较得到的，这个频段貌似效果最好，不过其他也可尝试
        # index = get_first_index(meta, column_index=1, target_value="1test") # number of training trials
        train_data = np.r_[feat_map[:test_start], feat_map[test_start+test_length:]]
        train_labels = np.r_[labels[:test_start], labels[test_start+test_length:]]
        test_data = feat_map[test_start:test_start+test_length]
        test_labels = labels[test_start:test_start+test_length]
        print(train_data.shape, train_labels.shape, test_data.shape, test_labels)

    return train_data, test_data, train_labels, test_labels


if __name__ == '__main__':
    get_moabb_data_crs_sub("Weibo2014", 1, "mi")
    # train_data, test_data, train_labels, test_labels = get_moabb_data("2a", 1, 1, "mi")
    # print(train_data[0].shape)
