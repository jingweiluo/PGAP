import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy, gaussian_kde
from itertools import groupby
from scipy.spatial.distance import cdist
from collections import defaultdict
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

def sort_and_group(lst):
    return [item for item, group in groupby(sorted(lst)) for _ in group]

def ndToList(array):
    return [array[i] for i in range(array.shape[0])]

def knn_mixing_ratio(train_data, train_labels, k, neighbor_k):
    """
    计算每个样本的k近邻类别混合度，返回混合度最高的k个样本索引

    参数:
    train_data: 训练数据（可多维，会自动展平）
    train_labels: 训练标签（支持字符串标签）
    k: 需要返回的高混合度样本数量
    neighbor_k: 计算混合度时的近邻个数

    返回:
    high_mixing_indices: 高混合度样本的索引列表
    """
    # 数据预处理
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X_train = train_data.reshape(train_data.shape[0], -1)  # 展平为二维

    # 标签编码为0/1
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)  # 转换为数值标签

    # 计算k近邻（排除样本自身）
    nbrs = NearestNeighbors(n_neighbors=neighbor_k+1, algorithm='auto').fit(X_train)
    _, indices = nbrs.kneighbors(X_train)  # indices包含自身索引

    # 混合度计算
    mixing_scores = []
    for i in range(len(X_train)):
        neighbor_labels = y_train[indices[i][1:]]  # 排除第一个元素（自身）
        current_label = y_train[i]
        # 计算异类邻居占比
        mixing_ratio = np.sum(neighbor_labels != current_label) / neighbor_k
        mixing_scores.append(mixing_ratio)

    # 选择混合度最高的k个样本
    mixing_scores = np.array(mixing_scores)
    high_mixing_indices = np.argsort(-mixing_scores)[:k]  # 降序排序取前k（高混合度）
    low_mixing_indices = np.argsort(mixing_scores)[:k]  # 升序排序取前k（低混合度）

    return ndToList(low_mixing_indices)  # 假设ndToList将numpy数组转为list

def lr(train_data, train_labels, k, filter_k=10):
    """
    输入：
    - train_data: 特征数组，形状为 (n_samples, n_features) 或更高维
    - train_labels: 多类别标签，shape 为 (n_samples,)
    - k: 最终返回的样本总数，尽量类别均衡
    - filter_k: 邻域一致性判断的近邻数（默认10）

    返回：
    - easiest_indices: 符合熵小 + 邻域一致性的样本索引，类别平衡，总数约为 k
    """
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X = train_data.reshape(train_data.shape[0], -1)
    y = train_labels

    # 标签编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    categories = np.unique(y_encoded)
    n_categories = len(categories)
    samples_per_class = k // n_categories

    if k % n_categories != 0:
        print(f"⚠️ 警告: k={k} 不能被类别数 {n_categories} 整除，最后将略少于 k 个样本")

    # 训练逻辑回归模型
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y_encoded)
    probas = model.predict_proba(X)
    sample_entropies = entropy(probas.T, base=2)

    # 计算距离矩阵（用于最近邻过滤）
    all_dists = pairwise_distances(X, X)

    easiest_indices = []

    for cat in categories:
        cat_indices = np.where(y_encoded == cat)[0]
        cat_entropies = sample_entropies[cat_indices]

        # 对该类按熵升序排序（熵越小越“确定”）
        sorted_cat_indices = cat_indices[np.argsort(cat_entropies)]

        # 贪心挑选满足邻域一致性的点
        selected_cat = []
        ptr = 0
        while len(selected_cat) < samples_per_class and ptr < len(sorted_cat_indices):
            idx = sorted_cat_indices[ptr]
            ptr += 1

            dists = all_dists[idx].copy()
            dists[idx] = np.inf
            neighbor_ids = np.argsort(dists)[:filter_k]
            neighbor_labels = y_encoded[neighbor_ids]
            same_label_count = np.sum(neighbor_labels == y_encoded[idx])

            if same_label_count > filter_k / 2:
                selected_cat.append(idx)

        if len(selected_cat) < samples_per_class:
            print(f"⚠️ 类别 {cat} 中仅找到 {len(selected_cat)} 个符合条件的样本，目标是 {samples_per_class}")

        easiest_indices.extend(selected_cat)

    return (easiest_indices)

def svm(train_data, train_labels, k):
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X_train = train_data.reshape(train_data.shape[0], -1)

    # 将字符串标签转换为数值（0, 1）
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    # 训练SVM模型
    # svm = SVC(kernel='rbf', C=1.0, gamma='scale')  # 线性核
    svm = SVC(kernel='linear', C=1.0)  # 线性核
    svm.fit(X_train, y_train)

    # 获取支持向量及其索引
    support_vectors = svm.support_vectors_
    support_indices = svm.support_  # 支持向量的原始索引
    support_labels = y_train[support_indices]  # 获取支持向量对应的类别

    # 计算每个样本的决策边界距离（决策函数值）
    decision_values = svm.decision_function(X_train)

    # 计算不确定度（取绝对值，越接近0越不确定）
    uncertainty_scores = np.abs(decision_values)

    # 选择最不确定的 k 个样本索引
    most_uncertain_indices = np.argsort(uncertainty_scores)[:k]
    most_certain_indices = np.argsort(-uncertainty_scores)[:k]
    final_indices = np.concatenate((most_certain_indices, most_uncertain_indices), axis=0)
    return sort_and_group(ndToList(most_certain_indices))

def balanced_svm(train_data, train_labels, k, filter_k=5):
    '''
    filter_k: 找到filter_k个近邻并检查是否有超过半数是同类样本
    '''
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X_train = train_data.reshape(train_data.shape[0], -1)

    # 标签编码
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    class_labels = np.unique(y_train)
    n_classes = len(class_labels)

    # 每类目标样本数（只取确定样本）
    k_per_class = k // n_classes

    # 拟合 SVM
    svm = SVC(kernel='linear', C=1.0, decision_function_shape='ovr')
    svm.fit(X_train, y_train)
    decision_values = svm.decision_function(X_train)

    if len(decision_values.shape) == 1:
        uncertainty_scores = np.abs(decision_values)
    else:
        top2 = np.partition(-np.abs(decision_values), 1, axis=1)[:, :2]
        uncertainty_scores = -np.abs(top2[:, 0] - top2[:, 1])

    # 距离矩阵
    dists = pairwise_distances(X_train, X_train)

    def filter_neighbors(sorted_indices, label, needed):
        selected = []
        ptr = 0
        while len(selected) < needed and ptr < len(sorted_indices):
            idx = sorted_indices[ptr]
            ptr += 1
            dist_vec = dists[idx].copy()
            dist_vec[idx] = np.inf
            neighbors = np.argsort(dist_vec)[:filter_k]
            neighbor_labels = y_train[neighbors]
            same_label_count = np.sum(neighbor_labels == label)
            if same_label_count >= filter_k * 0.7:
                selected.append(idx)
        return selected

    final_indices = []

    for cls in class_labels:
        cls_indices = np.where(y_train == cls)[0]
        cls_uncertainties = uncertainty_scores[cls_indices]
        sorted_indices = cls_indices[np.argsort(cls_uncertainties)]

        # 只选确定 + 邻域一致性样本（从后往前）
        certain_selected = filter_neighbors(sorted_indices[::-1], cls, k_per_class)

        if len(certain_selected) < k_per_class:
            print(f"⚠️ 类别 {cls} 的“确定+纯净”样本不足：{len(certain_selected)} / {k_per_class}")

        final_indices.extend(certain_selected)

    return final_indices

def svm_near(train_data, train_labels, k):
    """
    选出距离 SVM 决策边界最近的 k 个样本，类别间尽可能平衡。

    参数：
    ----------
    train_data : array-like
        特征数据，形状为 (N, ...)。
    train_labels : array-like
        标签数据，长度为 N。
    k : int
        要选出的总样本数（需能被类别数整除）。

    返回：
    ----------
    final_indices : list
        被选中样本的索引。
    """
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X_train = train_data.reshape(train_data.shape[0], -1)

    # 标签编码
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    class_labels = np.unique(y_train)
    n_classes = len(class_labels)

    assert k % n_classes == 0, "k 必须能被类别数整除以保证均衡"
    k_per_class = k // n_classes

    # 拟合 SVM
    svm = SVC(kernel='linear', C=1.0, decision_function_shape='ovr')
    svm.fit(X_train, y_train)
    decision_values = svm.decision_function(X_train)

    # 计算每个样本距离决策边界的“置信度”
    if len(decision_values.shape) == 1:
        # 二分类情况
        uncertainty_scores = np.abs(decision_values)
    else:
        # 多分类，取置信度前两名的差
        top2 = np.partition(-np.abs(decision_values), 1, axis=1)[:, :2]
        uncertainty_scores = np.abs(top2[:, 0] - top2[:, 1])

    # 越小表示越靠近决策边界
    final_indices = []

    for cls in class_labels:
        cls_indices = np.where(y_train == cls)[0]
        cls_uncertainty = uncertainty_scores[cls_indices]

        # 取距离决策边界最近的前 k_per_class 个样本
        sorted_indices = cls_indices[np.argsort(cls_uncertainty)[:k_per_class]]
        final_indices.extend(sorted_indices)

    return final_indices

def balanced_density_selection(train_data, train_labels, k):
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X_train = train_data.reshape(train_data.shape[0], -1)

    # 将字符串标签转换为数值（0, 1）
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    # 分别获取正类和负类的索引
    pos_indices = np.where(y_train == 1)[0]
    neg_indices = np.where(y_train == 0)[0]

    # 计算密度
    def compute_density(indices):
        subset = X_train[indices]
        kde = gaussian_kde(subset.T)  # KDE 计算密度
        densities = kde(subset.T)  # 获取密度值
        return densities

    pos_densities = compute_density(pos_indices)
    neg_densities = compute_density(neg_indices)

    # 选择密度最高的 k//2 个样本
    k_half = k // 2
    top_pos_indices = pos_indices[np.argsort(-pos_densities)[:k_half]]
    top_neg_indices = neg_indices[np.argsort(-neg_densities)[:k_half]]

    # 合并索引
    most_dense_indices = np.concatenate([top_pos_indices, top_neg_indices])

    return most_dense_indices.tolist()

def find_hardest_samples_rf(train_data, train_labels, k):
    """
    用随机森林计算熵，并返回熵最大的 k 个样本索引
    """
    X = train_data.reshape(train_data.shape[0], -1)

    # 将字符串标签转换为数值（0, 1）
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_labels)

    # 训练随机森林模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 获取类别概率
    probas = model.predict_proba(X)  # 形状 (n_samples, 2)

    # 计算熵 H(x) = -sum(p * log p)
    sample_entropies = entropy(probas.T, base=2)  # 计算每个样本的熵

    # 按熵值从大到小排序
    sorted_indices = np.argsort(sample_entropies)[::-1]

    # 选取熵最大的 k 个样本索引
    hardest_indices = sorted_indices[:k]
    # print("RF选出的熵最大的 k 个样本:", hardest_indices, train_labels[hardest_indices], y[hardest_indices])

    return hardest_indices

def simn(train_data, train_labels, n, k=10):
    # 转为 numpy array 并展开到二维 (n_samples, n_features)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X_train = train_data.reshape(train_data.shape[0], -1)

    # 将字符串标签转换为数值标签（例如 0/1/2/...）
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    # 记录每个样本“同标签邻居比例”
    ratio_list = []

    for i in range(len(X_train)):
        # 计算当前样本与其他所有样本的欧式距离
        dists = np.linalg.norm(X_train - X_train[i], axis=1)
        # 避免把自己算进 k 个最近邻，把自身距离设为无穷大
        dists[i] = float('inf')

        # 找到距离最小的 k 个邻居
        nn_indices = np.argsort(dists)[:k]

        # 统计最近 k 个邻居里，与当前样本标签相同的数量
        same_label_count = np.sum(y_train[nn_indices] == y_train[i])

        # 计算“同标签占比”
        same_label_ratio = same_label_count / k

        ratio_list.append((i, same_label_ratio))

    # 根据同标签占比从高到低排序
    ratio_list.sort(key=lambda x: x[1], reverse=True)

    # 取出前 n 个样本索引
    top_n_indices = [idx for idx, ratio in ratio_list[:n]]

    return sort_and_group(top_n_indices)

def bal_simn(train_data, train_labels, n, k=10):
    """
    从数据集中选出同类纯度最高的样本，但最后保证输出中
    一半样本来自类别0, 一半来自类别1。

    参数：
    -----------
    train_data : array-like
        训练数据，形状可为 (num_samples, ...)，可以是图像或其他特征形式。
    train_labels : array-like
        训练标签，与 train_data 对应。只适用于二分类场景。
    n : int
        最终选出的样本总数。假设 n 是偶数。
    k : int
        用来计算同类纯度的近邻个数。

    返回：
    -----------
    top_n_indices : list
        根据同类纯度排序后，取出类别均衡的前 n 个样本索引。
    """
    # 转为 numpy array 并展开到二维 (n_samples, n_features)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X_train = train_data.reshape(train_data.shape[0], -1)

    # 将字符串标签转换为数值标签（例如 0/1）
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    # 确保只有两类(0/1)
    unique_labels = np.unique(y_train)
    assert len(unique_labels) == 2, "该方法仅适用于二分类场景。"

    # 存储 (样本索引, 同类纯度, 该样本标签)
    ratio_list = []

    for i in range(len(X_train)):
        # 计算当前样本与其他所有样本的欧式距离
        dists = np.linalg.norm(X_train - X_train[i], axis=1)
        # 避免把自己算进 k 个最近邻，把自身距离设为无穷大
        dists[i] = float('inf')

        # 找到距离最小的 k 个邻居
        nn_indices = np.argsort(dists)[:k]

        # 统计这些邻居中，标签与当前样本相同的数量
        same_label_count = np.sum(y_train[nn_indices] == y_train[i])
        # 计算同类纯度
        same_label_ratio = same_label_count / k

        ratio_list.append((i, same_label_ratio, y_train[i]))

    # 我们要在选出样本时保证两类平衡，所以先把 ratio_list 分成两部分
    ratio_list_label0 = [item for item in ratio_list if item[2] == 0]
    ratio_list_label1 = [item for item in ratio_list if item[2] == 1]

    # 在各自类别中按照 same_label_ratio 从高到低排序
    ratio_list_label0.sort(key=lambda x: x[1], reverse=True)
    ratio_list_label1.sort(key=lambda x: x[1], reverse=True)

    # 假设 n 是偶数，n/2来自类别0，n/2来自类别1
    n_half = n // 2
    # 如果某一类数量不足 n/2，则可以自己决定是直接取完该类全部，还是报错
    # 这里简单写成直接取最前 n_half
    top_n_indices_label0 = [x[0] for x in ratio_list_label0[:n_half]]
    top_n_indices_label1 = [x[0] for x in ratio_list_label1[:n_half]]

    # 合并
    top_n_indices = top_n_indices_label0 + top_n_indices_label1

    return top_n_indices

def bal_simn_multi(train_data, train_labels, n, num_cls, k=18, measurement='ed'):
    """
    从数据集中选出同类纯度最高的样本，最终保证输出中各类别样本数量均衡。

    参数：
    -----------
    train_data : array-like
        训练数据，形状可为 (num_samples, ...)，可以是图像或其他特征形式。
    train_labels : array-like
        训练标签，与 train_data 对应。适用于多分类场景。
    n : int
        最终选出的样本总数。确保 n 能被 num_cls 整除。
    num_cls : int
        类别数量。
    k : int
        用来计算同类纯度的近邻个数。

    返回：
    -----------
    top_n_indices : list
        根据同类纯度排序后，取出类别均衡的前 n 个样本索引。
    """
    # 1. 确保 n 能够被 num_cls 整除，否则无法均衡选择
    assert n % num_cls == 0, "n 必须能被 num_cls 整除，以保证类别均衡"
    k = len(train_data) // 8

    # 2. 转换数据格式
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X_train = train_data.reshape(train_data.shape[0], -1)

    # 3. 将字符串标签转换为数值标签（如 0, 1, 2,..., num_cls-1）
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    # 4. 获取所有类别
    unique_labels = np.unique(y_train)
    assert len(unique_labels) == num_cls, f"数据中仅包含 {len(unique_labels)} 类，但期望 {num_cls} 类"

    # 5. 计算每个样本的同类纯度
    ratio_list = []  # 存储 (样本索引, 同类纯度, 该样本类别)

    for i in range(len(X_train)):
        if measurement == 'ed':
            dists = np.linalg.norm(X_train - X_train[i], axis=1)
            dists[i] = float('inf')
            nn_indices = np.argsort(dists)[:k]
        elif measurement == 'cs':  # 'cs' 余弦相似度
            sims = cosine_similarity(X_train[i:i+1], X_train).flatten()
            sims[i] = -1  # 避免把自己算进去
            nn_indices = np.argsort(sims)[-k:]  # 越大相似度越高

        # 计算同类邻居的比例
        same_label_count = np.sum(y_train[nn_indices] == y_train[i])
        same_label_ratio = same_label_count / k

        ratio_list.append((i, same_label_ratio, y_train[i]))

    # 6. 按类别分组，并根据同类纯度排序
    ratio_dict = {cls: [] for cls in unique_labels}  # {类别: [(索引, 纯度, 类别), ...]}

    for item in ratio_list:
        ratio_dict[item[2]].append(item)

    for cls in unique_labels:
        ratio_dict[cls].sort(key=lambda x: x[1], reverse=True)  # 按同类纯度降序

    # 7. 选取每个类别 `n // num_cls` 个样本
    num_per_class = n // num_cls
    top_n_indices = []

    for cls in unique_labels:
        class_samples = ratio_dict[cls][:num_per_class]
        top_n_indices.extend([x[0] for x in class_samples])

    return top_n_indices

def bal_simn_greedy_multi(train_data, train_labels, n, k=10, purity_threshold=0.8):
    """
    多类版本：从数据集中选出同类纯度高且分布多样的样本，确保类别均衡。

    参数：
    - train_data: 输入特征，形状为 (num_samples, ...)
    - train_labels: 标签，任意多个类别
    - n: 最终要选出的总样本数（尽量让每类占 n / num_classes 个）
    - k: 计算近邻纯度时的邻居数
    - purity_threshold: 近邻中同类比例需大于此阈值，才参与候选

    返回：
    - top_n_indices: 类别平衡 + 贪心采样后的样本索引列表
    """
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X_train = train_data.reshape(train_data.shape[0], -1)

    # 标签编码为整数
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    classes = np.unique(y_train)
    num_classes = len(classes)
    per_class_n = n // num_classes

    # 构建候选样本池（满足纯度要求）
    ratio_dict = defaultdict(list)  # 每个类别一个列表

    for i in range(len(X_train)):
        dists = np.linalg.norm(X_train - X_train[i], axis=1)
        dists[i] = float('inf')
        nn_indices = np.argsort(dists)[:k]
        same_label_count = np.sum(y_train[nn_indices] == y_train[i])
        same_label_ratio = same_label_count / k

        if same_label_ratio >= purity_threshold:
            label = y_train[i]
            ratio_dict[label].append((i, same_label_ratio))

    # 贪心采样：保持每类样本多样性
    def greedy_sampling(indices_with_purity, num_samples):
        if len(indices_with_purity) <= num_samples:
            return [x[0] for x in indices_with_purity]

        sorted_list = sorted(indices_with_purity, key=lambda x: x[1], reverse=True)
        selected_indices = [sorted_list[0][0]]
        selected_features = [X_train[selected_indices[0]].reshape(1, -1)]

        for _ in range(num_samples - 1):
            remaining = [x for x in sorted_list if x[0] not in selected_indices]
            if not remaining:
                break
            remaining_indices = np.array([x[0] for x in remaining])
            remaining_features = X_train[remaining_indices]
            min_dists = np.min(cdist(remaining_features, np.vstack(selected_features)), axis=1)
            farthest_idx = remaining[np.argmax(min_dists)][0]
            selected_indices.append(farthest_idx)
            selected_features.append(X_train[farthest_idx].reshape(1, -1))
        return selected_indices

    # 遍历每个类，执行贪心采样
    top_n_indices = []
    for cls in classes:
        cls_samples = ratio_dict.get(cls, [])
        selected = greedy_sampling(cls_samples, per_class_n)
        top_n_indices.extend(selected)

    return top_n_indices

def soft_bal_simn(train_data, train_labels, n, k=10):
    """
    计算所有样本的“同类纯度”，按纯度从高到低排序，并返回前 n 个。
    但若这 n 个全是同一类别，则强制至少包含另一类的样本（用纯度最高的那一个替换）。

    参数：
    -----------
    train_data : array-like
        训练数据，形状 (num_samples, ...)。可以是图像或其他形式。
    train_labels : array-like
        训练标签，与 train_data 对应。只适用于二分类场景。
    n : int
        最终选出的样本数量。
    k : int
        用来计算同类纯度的近邻个数。

    返回：
    -----------
    final_indices : list
        长度为 n 的样本索引列表，按照纯度从高到低排列，且至少包含两个不同类别。
    """
    # 转为 numpy array 并展开到二维 (n_samples, n_features)
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X_train = train_data.reshape(train_data.shape[0], -1)

    # 将字符串标签转换为数值标签（例如 0/1）
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    # 确保只有两类(0/1)
    unique_labels = np.unique(y_train)
    assert len(unique_labels) == 2, "该方法仅适用于二分类场景。"

    # 存储 (样本索引, 同类纯度, 该样本标签)
    ratio_list = []

    for i in range(len(X_train)):
        # 计算当前样本与其他所有样本的欧式距离
        dists = np.linalg.norm(X_train - X_train[i], axis=1)
        # 避免把自己算进 k 个最近邻，把自身距离设为无穷大
        dists[i] = float('inf')

        # 找到距离最小的 k 个邻居
        nn_indices = np.argsort(dists)[:k]

        # 统计这些邻居中，标签与当前样本相同的数量
        same_label_count = np.sum(y_train[nn_indices] == y_train[i])
        # 计算同类纯度
        same_label_ratio = same_label_count / k

        ratio_list.append((i, same_label_ratio, y_train[i]))

    # 按纯度从高到低排序
    ratio_list.sort(key=lambda x: x[1], reverse=True)

    # 先取前 n 个
    top_n = ratio_list[:n]
    labels_in_top_n = {x[2] for x in top_n}

    if len(labels_in_top_n) == 2:
        # 已经有两个类别，直接返回
        final_indices = [x[0] for x in top_n]
        return final_indices
    else:
        # 如果只有一种类别，找出缺失的类别
        full_label_set = set([0, 1])
        existing_label = list(labels_in_top_n)[0]
        missing_label = list(full_label_set - {existing_label})[0]

        # 在其余样本里(从第 n 个开始) 按纯度从高到低找该缺失类别
        for candidate in ratio_list[n:]:
            if candidate[2] == missing_label:
                # 找到一个缺失类别的样本
                # 用它替换掉 top_n 里纯度最小的那个（即 top_n[-1]）
                top_n[-1] = candidate
                # 再按纯度从高到低重新排序
                top_n.sort(key=lambda x: x[1], reverse=True)

                # 此时至少有了两个类别，退出循环
                break
        else:
            # 如果循环走完也没找到缺失类别，说明数据里本身就不包含该类别
            # 根据需求可选择抛出异常、或返回原 top_n
            print("警告：数据集中根本不存在另一类别，无法实现两类共存。")
            # 这里就直接返回原 top_n
            # 也可 raise ValueError("No sample of the missing label in the dataset.")

        # 最终结果
        final_indices = [x[0] for x in top_n]
        return final_indices

def bal_intra_class_min_dist(train_data, train_labels, n, k=10, m=10, alpha=0.5, distance_metric="euclidean"):
    """
    从训练集中选择 n 个样本（类别均衡），每类选择得分最低的样本：
    - 同类邻居距离小（亲和性）
    - 异类邻居距离大（分离性）

    参数：
    ----------
    - n: 最终总共选择的样本数，需能被类别数整除
    - k: 计算同类最近邻数
    - m: 计算异类最近邻数
    - alpha: 0~1 加权系数（越大越偏重同类距离）
    - distance_metric: "euclidean" 或 "cosine"
    """
    k = len(train_data) // 8
    m = len(train_data) // 8

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X_train = train_data.reshape(train_data.shape[0], -1)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)

    unique_labels = np.unique(y_train)
    num_cls = len(unique_labels)
    assert n % num_cls == 0, "n 必须能被类别数整除以实现类别均衡"
    num_per_class = n // num_cls

    # 选择距离函数
    if distance_metric == "cosine":
        dist_func = cosine_distances
    elif distance_metric == "euclidean":
        dist_func = euclidean_distances
    else:
        raise ValueError("Unsupported distance metric. Use 'euclidean' or 'cosine'.")

    selected_indices = []

    for cls in unique_labels:
        cls_indices = np.where(y_train == cls)[0]
        other_indices = np.where(y_train != cls)[0]
        cls_vectors = X_train[cls_indices]
        other_vectors = X_train[other_indices]

        dist_list = []

        for i, idx in enumerate(cls_indices):
            xi = X_train[idx]

            # === 同类部分 ===
            if len(cls_indices) <= k:
                continue
            cls_neighbors = np.delete(cls_vectors, i, axis=0)
            intra_dists = dist_func(xi.reshape(1, -1), cls_neighbors).flatten()
            top_k_intra = np.sort(intra_dists)[:k]
            intra_score = np.mean(top_k_intra)

            # === 异类部分 ===
            if len(other_vectors) < m:
                continue
            inter_dists = dist_func(xi.reshape(1, -1), other_vectors).flatten()
            top_m_inter = np.sort(inter_dists)[:m]
            inter_score = np.mean(top_m_inter)

            # === 综合评分（越小越好）===
            score = alpha * intra_score - (1 - alpha) * inter_score
            dist_list.append((idx, score))

        # 按综合得分排序，选前 num_per_class 个
        dist_list.sort(key=lambda x: x[1])
        selected_cls = [x[0] for x in dist_list[:num_per_class]]
        selected_indices.extend(selected_cls)

    return selected_indices

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 2)  # 二分类输出 2 维

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # 输出 logits
        return x

def find_hardest_samples_nn(train_data, train_labels, k, epochs=10, batch_size=32):
    """
    用神经网络计算熵，并返回熵最大的 k 个样本索引
    """
    X = train_data.reshape(train_data.shape[0], -1)
    # 将字符串标签转换为数值（0, 1）
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_labels)

    # 转换为 PyTorch 张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # 训练神经网络
    input_dim = X.shape[1]
    model = SimpleNN(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 训练数据集
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练模型
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # 计算每个样本的预测概率
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probas = torch.softmax(logits, dim=1).numpy()  # (n_samples, 2)

    # 计算熵
    sample_entropies = entropy(probas.T, base=2)

    # 按熵值从大到小排序
    sorted_indices = np.argsort(sample_entropies)[::-1]

    # 选取熵最大的 k 个样本索引
    hardest_indices = sorted_indices[:k]
    # print("NN选出的熵最大的 k 个样本:", hardest_indices, train_labels[hardest_indices])

    return hardest_indices

if __name__ == '__main__':
    import sys
    import os
    current_file_path = os.path.abspath(__file__)
    parent_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    sys.path.append(parent_parent_dir)
    from active_prompt.load.load_moabb import get_moabb_data

    dataset_name = "BNCI2015_001" # BNCI2015_001
    sub_index = 3
    test_id = 1
    k = 4
    para_dict = {
        '2a': 'mi',
        '2b': 'mi',
        'BNCI2015_001': 'mi',
        'BNCI2014_002': 'mi',
        'Kalunga2016': 'ssvep',
    }

    train_data, test_data, train_labels, test_labels = get_moabb_data(dataset_name, sub_index, test_id, para_dict[dataset_name]) # 2b数据集，sub_index, test_id
    svm_list = svm(train_data, train_labels, k)
    lr(train_data, train_labels, k)
    # balanced_svm(train_data, train_labels, k)
    # find_hardest_samples_rf(train_data, train_labels, k)
    # find_hardest_samples_nn(train_data, train_labels, k, epochs=10, batch_size=32)
