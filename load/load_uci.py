import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler  # 添加标准化模块

def get_uci_data_cv(dataset_name, sub_index=1, para='uci', n_splits=5):
    uci_path = '/data1/labram_data/UCI/'
    data_path = uci_path + dataset_name

    # 读取数据
    df = pd.read_csv(data_path + f'/{dataset_name}.data', header=None)

    # 添加列名（可选）
    # df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

    df = df.dropna()  # 删除可能存在的空行

    # 拆分特征和标签
    if dataset_name == 'iris':
        X = df.iloc[:, :-1].to_numpy(dtype=np.float32)  # N x F 特征数组
        labels = df.iloc[:, -1].astype(str).to_numpy()  # N 维标签字符串数组
    elif dataset_name == 'wine':
        X = df.iloc[:, 1:].to_numpy(dtype=np.float32)  # N x F 特征数组
        labels = df.iloc[:, 0].astype(str).to_numpy()  # N 维标签字符串数组
    elif dataset_name == 'wdbc':
        X = df.iloc[:, 2:].to_numpy(dtype=np.float32)  # N x F 特征数组
        labels = df.iloc[:, 1].astype(str).to_numpy()  # N 维标签字符串数组
    elif dataset_name == 'glass':
        X = df.iloc[:, 1:-1].to_numpy(dtype=np.float32)  # N x F 特征数组
        labels = df.iloc[:, -1].astype(str).to_numpy()  # N 维标签字符串数组

    # 显示形状作为确认
    # print(X.shape, y.shape)


    # k-folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, labels)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # # ✅ 添加归一化处理（fit on train, transform on both）
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train)
        # X_test = scaler.transform(X_test)

        print(f"[Fold {fold_idx}] train: {X_train.shape}, test: {X_test.shape}")
        yield X_train, X_test, y_train, y_test
