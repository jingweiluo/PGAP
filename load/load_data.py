import scipy.io # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import mne # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

datapath_SEED_IV_eeg_feature = '/data1/labram_data/SEED_IV/eeg_feature_smooth/'
session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
all_labels = session1_label + session2_label + session3_label
all_labels = [str(x) for x in all_labels]

def resample_trial(trial, target_len=42):
    """
    Resample a trial with shape (6, T, 5) to shape (6, target_len, 5)
    """
    C, T, F = trial.shape
    reshaped = trial.transpose(1, 0, 2).reshape(T, -1)  # (T, 6*5)
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x_old, reshaped, axis=0, kind='linear', fill_value='extrapolate')
    resampled = f(x_new)  # (target_len, 30)
    final = resampled.reshape(target_len, C, F).transpose(1, 0, 2)  # (6, 42, 5)
    return final

def extract_statistical_features(trial_data):
    """
    输入形状：(channels, time_steps, freq_bands)
    对每个 trial 提取多个统计特征：mean, std, max, min, median
    返回 shape: (channels × freq_bands × num_stats,)
    """
    trial_data = trial_data[:2, :, :4] # 只取2个通道，4个频段
    mean_feat = np.mean(trial_data, axis=1)   # shape: (channels, freq_bands)
    std_feat = np.std(trial_data, axis=1)
    max_feat = np.max(trial_data, axis=1)
    min_feat = np.min(trial_data, axis=1)
    median_feat = np.median(trial_data, axis=1)

    features = np.concatenate([
        mean_feat, std_feat#, max_feat, min_feat, median_feat
    ], axis=1)  # shape: (channels, freq_bands × 5)

    return features.flatten()  # shape: (channels × freq_bands × 5,)

def get_seed_data(subject_index, pca_dim=10, target_len=42):
    all_trials = []

    # 1. 读取全部72个trial
    for session in range(1, 4):
        eeg_datafile = datapath_SEED_IV_eeg_feature + str(subject_index) + '_' + str(session) + '.mat'
        eeg_data = scipy.io.loadmat(eeg_datafile)

        for i in range(1, 24 + 1):
            eeg_feature_name = 'de_LDS' + str(i)
            eeg_trial = eeg_data[eeg_feature_name]  # shape (62, 42, 5)
            selected_chans_trial = eeg_trial[[14, 22, 23, 31, 32, 40], :, :]  # shape (6, ?, 5) 中间的特征值数量不一样
            # final_trial = resample_trial(selected_chans_trial, target_len=target_len)
            final_trial = extract_statistical_features(selected_chans_trial)
            print(final_trial.shape)
            all_trials.append(final_trial.reshape(-1))  # flatten to 1D)

    all_trials = np.array(all_trials)  # shape: (72, 1260)

    # 2. 划分训练集（前48个）和测试集（后24个）
    X_train = all_trials[:48]
    X_test = all_trials[48:]

    # 3. 在训练集上fit PCA
    pca = PCA(n_components=pca_dim)
    train_data = pca.fit_transform(X_train)

    # 4. 用相同PCA变换测试集
    test_data = pca.transform(X_test)

    y = np.array(all_labels)

    train_labels, test_labels = y[:48], y[48:72]
    return train_data, test_data, train_labels, test_labels

def load_all_segs(subject_index):
    all_trials = []
    all_segs = []
    all_segs_labels = []

    for session in range(1,4):
        # load subject data
        eeg_datafile = datapath_SEED_IV_eeg_feature + str(subject_index) + '_' + str(session) + '.mat'
        eeg_data = scipy.io.loadmat(eeg_datafile)

        for i in range(1, 24+1):
            eeg_feature_name = 'de_LDS' + str(i)
            eeg_trial = eeg_data[eeg_feature_name]
            selected_chans_trial = eeg_trial[[14, 22, 23, 31, 32, 40], :, :] # 选择指定的6个通道
            all_trials.append(selected_chans_trial)

    for (i, tri) in enumerate(all_trials):
        for j in range(tri.shape[1]):
            all_segs.append(tri[:, j, :])
            all_segs_labels.append(all_labels[i])

    train_data, test_data, train_labels, test_labels = train_test_split(all_segs, all_segs_labels, test_size=0.2, random_state=42)
    return train_data, test_data, train_labels, test_labels

if __name__ == '__main__':
    train_data, test_data, train_labels, test_labels = get_seed_data(1)
    print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)

# load_bci_iv_2b(9)