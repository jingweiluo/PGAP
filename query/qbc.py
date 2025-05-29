import sys
import os
current_file_path = os.path.abspath(__file__)
parent_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
sys.path.append(parent_parent_dir)
import numpy as np
# RANDOM_STATE_SEED = 1
# np.random.seed(RANDOM_STATE_SEED)
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from active_prompt.load.load_moabb import get_moabb_data
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from modAL.models import ActiveLearner, Committee
from modAL.disagreement import vote_entropy_sampling, max_disagreement_sampling, consensus_entropy_sampling
from scipy.stats import entropy
from copy import deepcopy
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

def findIndex(X, v):
    # np.all() 检查每行是否与目标向量完全匹配
    matches = np.all(X == v, axis=1)
    # np.where() 返回匹配项的索引
    matched_indices = np.where(matches)[0]

    # 检查是否找到了匹配项
    if matched_indices.size > 0:
        return matched_indices[0]  # 返回第一个匹配项的索引
    else:
        print("未在X中找到匹配的向量v")
        return None

def findNearestSampleIndex(cluster_samples, cluster_centroids, measurement):
    # 返回在聚类中的local index
    if measurement == "ed":
        distances = np.linalg.norm(cluster_samples - cluster_centroids, axis=1)
        nearest_local_index = np.argmin(distances)
    elif measurement == "cs":
        # 计算余弦相似度
        dot_products = np.dot(cluster_samples, cluster_centroids)
        norms_samples = np.linalg.norm(cluster_samples, axis=1)
        norm_centroid = np.linalg.norm(cluster_centroids)
        cosine_similarities = dot_products / (norms_samples * norm_centroid)
        nearest_local_index = np.argmax(cosine_similarities)
    else:
        raise ValueError("Unsupported measurement type. Use 'ed' for Euclidean distance or 'cs' for cosine similarity.")
    return nearest_local_index

def findNearestKSampleIndices(cluster_samples, cluster_centroids, measurement, k=1):
    """
    返回在聚类中距离中心最近的k个样本的local indices。

    参数:
    cluster_samples : array-like, shape (n_samples, n_features)
        聚类中的样本点。
    cluster_centroids : array-like, shape (n_features,)
        聚类的中心点。
    measurement : str
        用于度量距离的类型，可以是"ed"（欧几里得距离）或"cs"（余弦相似度）。
    k : int
        返回的最近样本点的数量。

    返回:
    nearest_k_indices : array, shape (k,)
        最近的k个样本点的indices。
    """
    if measurement == "ed":
        distances = np.linalg.norm(cluster_samples - cluster_centroids, axis=1)
        nearest_k_indices = np.argsort(distances)[:k]
    elif measurement == "cs":
        # 计算余弦相似度
        dot_products = np.dot(cluster_samples, cluster_centroids)
        norms_samples = np.linalg.norm(cluster_samples, axis=1)
        norm_centroid = np.linalg.norm(cluster_centroids)
        cosine_similarities = dot_products / (norms_samples * norm_centroid)
        nearest_k_indices = np.argsort(cosine_similarities)[-k:][::-1]
    else:
        raise ValueError("Unsupported measurement type. Use 'ed' for Euclidean distance or 'cs' for cosine similarity.")

    return nearest_k_indices

def find_init_centroids(X, init_num, measurement):
    '''
    返回k个cluster中，离质心最近的样本点的index
    '''
    kmeans = KMeans(init_num, random_state=42)
    kmeans.fit(X)
    # 获取初始质心
    centroids = kmeans.cluster_centers_
    # 为每个数据点分配聚类标签
    labels = kmeans.labels_

    cluster_map = {i : [] for i in range(centroids.shape[0])}
    for idx, label in enumerate(labels):
        cluster_map[label].append(idx)

    # 找到每个聚类中心最近的样本点
    nearest_indices = []
    for i, centroid in enumerate(centroids):
        cluster_samples = X[cluster_map[i]]
        # distances = np.linalg.norm(cluster_samples - centroid, axis=1)
        # nearest_index = np.argmin(distances)
        nearest_local_index = findNearestSampleIndex(cluster_samples, centroid, measurement)
        nearest_indices.append(cluster_map[i][nearest_local_index])
    return nearest_indices, kmeans

def find_largest_unlabeled_cluster(cur_labels, labeled_indices):
    """Find_largest_unlabeled_cluster

    Args:
        cur_labels (ndArray: (1, N)): cluster labels for all N trainning samples
        labeled_indices (list of int): labeled samples indices

    Returns:
        largest_cluster, max_size

    """
    # 创建一个集合，用于存储包含标记样本的簇
    labeled_clusters = set()

    # 遍历标记样本索引，识别这些样本所在的簇
    for idx in labeled_indices:
        labeled_clusters.add(cur_labels[idx])

    # cur_labels = np.array(cur_labels)
    # labeled_clusters = np.unique(cur_labels[labeled_indices])

    # 创建一个字典来统计不包含标记样本的簇的大小
    cluster_sizes = {}

    # 遍历所有样本的簇标签
    for i, label in enumerate(cur_labels):
        # 如果这个簇不包含任何标记样本，则计数
        if label not in labeled_clusters:
            if label in cluster_sizes:
                cluster_sizes[label] += 1
            else:
                cluster_sizes[label] = 1

    # 找出最大的簇
    largest_cluster = None
    max_size = -1
    for cluster, size in cluster_sizes.items():
        if size > max_size:
            max_size = size
            largest_cluster = cluster

    # 返回最大的簇的标签和大小
    return largest_cluster, max_size

def find_centroids(train_data, num_demos, measurement):
    train_data = np.array(train_data)
    X = train_data.reshape(train_data.shape[0],-1)

    nearest_indices, kmeans = find_init_centroids(X, num_demos, measurement)
    return nearest_indices

def kate_basic(train_data, train_labels, predict_sample, num_demos, measurement):
    """
    选择离predict最近的num_demos个示例
    """
    # ==========================================================初始化数据==================================================
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X = train_data.reshape(train_data.shape[0],-1)
    lab_map = {
        'left_hand': 1,
        'right_hand':2
    }
    y = np.array([lab_map[lab] for lab in train_labels])

    nearest_indices = findNearestKSampleIndices(X, predict_sample, measurement, k=num_demos)
    return nearest_indices

def kate(train_data, train_labels, predict_sample, num_demos, measurement):
    """
    先分成N个cluster，然后找到每个cluster中离待预测样本最近的点
    """
    # ==========================================================初始化数据==================================================
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X = train_data.reshape(train_data.shape[0],-1)
    lab_map = {
        'left_hand': 1,
        'right_hand':2
    }
    y = np.array([lab_map[lab] for lab in train_labels])

    # ==========================================================初始化数据==================================================
    kmeans = KMeans(num_demos, random_state=42)
    kmeans.fit(X)
    # 获取初始质心
    centroids = kmeans.cluster_centers_
    # 为每个数据点分配聚类标签
    labels = kmeans.labels_
    # 获取每个cluster包含的点的信息
    cluster_map = {i : [] for i in range(centroids.shape[0])}
    for idx, label in enumerate(labels):
        cluster_map[label].append(idx)

    # 找到每个聚类中心最近的样本点
    nearest_indices = []
    for i, centroid in enumerate(centroids):
        cluster_samples = X[cluster_map[i]]
        # distances = np.linalg.norm(cluster_samples - centroid, axis=1)
        # nearest_index = np.argmin(distances)
        nearest_local_index = findNearestSampleIndex(cluster_samples, predict_sample, measurement)
        nearest_indices.append(cluster_map[i][nearest_local_index])
    return nearest_indices

def kate_qbc(train_data, train_labels, predict_sample, num_demos, measurement, n_members=5, estimator=RandomForestClassifier):
    # 分成k个cluster，从每个cluster中找到一个距离最近的，一个qbc选择的，一共2k个
    # ==========================================================初始化数据==================================================
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X = train_data.reshape(train_data.shape[0],-1)
    lab_map = {
        'left_hand': 1,
        'right_hand':2
    }
    targets = np.array([lab_map[lab] for lab in train_labels])
    X_pool = deepcopy(X)
    y_pool = deepcopy(targets)

    # ==========================================================初始化数据==================================================
    kmeans = KMeans(num_demos // 2, random_state=42)
    kmeans.fit(X)
    # 获取初始质心
    centroids = kmeans.cluster_centers_
    # 为每个数据点分配聚类标签
    labels = kmeans.labels_
    # 获取每个cluster包含的点的信息
    cluster_map = {i : [] for i in range(centroids.shape[0])}
    for idx, label in enumerate(labels):
        cluster_map[label].append(idx)

    # 找到每个聚类中心最近的样本点
    nearest_indices = []
    for i, centroid in enumerate(centroids):
        cluster_samples = X[cluster_map[i]]
        nearest_local_index = findNearestSampleIndex(cluster_samples, predict_sample, measurement)
        nearest_indices.append(cluster_map[i][nearest_local_index])
    # ==========================================================初始化learner==================================================
    # 使用离predict最近的（而不是离centroid最近的）作为初始学习样本
    learner_list = list()

    for member_idx in range(n_members):
        # BootStrap 有放回的生成训练子集
        train_idx = np.random.choice(nearest_indices, size=num_demos // 2, replace=True)

        X_train = X[train_idx]
        y_train = targets[train_idx]

        # initializing learner
        learner = ActiveLearner(
            estimator=estimator(),
            X_training=X_train, y_training=y_train
        )
        learner_list.append(learner)

    # assembling the committee
    committee = Committee(
        learner_list=learner_list,
        query_strategy=consensus_entropy_sampling, # vote_entropy_sampling max_disagreement_sampling, consensus_entropy_sampling
    )

    # =============================================================Query=====================================================
    labeled_indices = deepcopy(nearest_indices) # 已经标注的sample list

    for i, centroid in enumerate(centroids):
        # 找到cluster中除了最近点以外点的indices
        updated_indices = np.where(cluster_map[i] != nearest_indices[i])[0]
        # 使用更新后的索引来选择X中的样本
        new_pool = X[updated_indices]

        query_idx, query_instance = committee.query(new_pool)
        gloabl_idx = findIndex(X, query_instance[0])
        committee.teach(
            X=X[[gloabl_idx]],
            y=targets[[gloabl_idx]]
        )
        labeled_indices.append(gloabl_idx)
    return labeled_indices, committee

def qbc(train_data, train_labels, n_members, n_initial, n_queries, estimator=RandomForestClassifier, init_idx=None, is_return_all=False):
    # ==========================================================初始化数据==================================================
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X = train_data.reshape(train_data.shape[0],-1)
    lab_map = {
        'left_hand': 1,
        'right_hand':2
    }
    targets = np.array([lab_map[lab] for lab in train_labels])
    X_pool = deepcopy(X)
    y_pool = deepcopy(targets)
    # ==========================================================初始化learner==================================================
    # n_members = 5
    learner_list = list()

    # n_initial = 12
    n_idx = init_idx if init_idx is not None else np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)

    for member_idx in range(n_members):
        train_idx = np.random.choice(n_idx, size=n_initial, replace=True)

        X_train = X_pool[train_idx]
        y_train = y_pool[train_idx]

        # initializing learner
        learner = ActiveLearner(
            estimator=estimator(),
            X_training=X_train, y_training=y_train
        )
        learner_list.append(learner)

    # creating a reduced copy of the data with the known instances removed
    X_pool = np.delete(X_pool, n_idx, axis=0)
    y_pool = np.delete(y_pool, n_idx)

    # assembling the committee
    committee = Committee(
        learner_list=learner_list,
        query_strategy=max_disagreement_sampling, # vote_entropy_sampling max_disagreement_sampling, consensus_entropy_sampling
    )

    # =============================================================Query=====================================================
    idx_list = []
    # n_queries = 8
    for _ in range(n_queries):
        query_idx, query_instance = committee.query(X_pool)
        gloabl_idx = findIndex(X, query_instance[0])
        idx_list.append(gloabl_idx)
        committee.teach(
            X=X_pool[query_idx],
            y=y_pool[query_idx]
        )
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)
    if is_return_all:
        return list(n_idx) + idx_list, committee
    else:
        return idx_list, committee

def rand_select(train_data, train_labels, n_members, n_initial, estimator):
    # ==========================================================初始化数据==================================================
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X = train_data.reshape(train_data.shape[0],-1)
    lab_map = {
        'left_hand': 1,
        'right_hand':2
    }
    targets = np.array([lab_map[lab] for lab in train_labels])
    X_pool = deepcopy(X)
    y_pool = deepcopy(targets)
    # ==========================================================初始化learner==================================================
    # n_members = 5
    learner_list = list()

    # n_initial = 20
    n_idx = np.random.choice(range(X_pool.shape[0]), size=n_initial, replace=False)

    for member_idx in range(n_members):
        train_idx = np.random.choice(n_idx, size=n_initial, replace=True)

        X_train = X_pool[train_idx]
        y_train = y_pool[train_idx]

        # initializing learner
        learner = ActiveLearner(
            estimator=estimator(),
            X_training=X_train, y_training=y_train
        )
        learner_list.append(learner)

    # creating a reduced copy of the data with the known instances removed
    X_pool = np.delete(X_pool, n_idx, axis=0)
    y_pool = np.delete(y_pool, n_idx)

    # assembling the committee
    committee = Committee(
        learner_list=learner_list,
        query_strategy=consensus_entropy_sampling, # vote_entropy_sampling max_disagreement_sampling, consensus_entropy_sampling
    )
    return n_idx, committee

def rd(query_method, train_data, train_labels, n_members, n_initial, n_queries, measurement, estimator, init_indices):
    # ==========================================================初始化数据==================================================
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    X = train_data.reshape(train_data.shape[0],-1)
    # lab_map = {
    #     'left_hand': 1,
    #     'right_hand':2
    # }
    # targets = np.array([lab_map[lab] for lab in train_labels])
    label_encoder = LabelEncoder()
    targets = label_encoder.fit_transform(train_labels)
    # ==========================================================初始化learner==================================================
    # n_members = 5
    learner_list = list()

    # n_initial = 12
    if init_indices:
        n_idx = init_indices
    else:
        n_idx, _ = find_init_centroids(X, n_initial, measurement)

    for member_idx in range(n_members):
        # BootStrap 有放回的生成训练子集
        train_idx = np.random.choice(n_idx, size=n_initial, replace=True)

        X_train = X[train_idx]
        y_train = targets[train_idx]

        # initializing learner
        learner = ActiveLearner(
            estimator=estimator(),
            X_training=X_train, y_training=y_train
        )
        learner_list.append(learner)

    # assembling the committee
    committee = Committee(
        learner_list=learner_list,
        query_strategy=max_disagreement_sampling, # vote_entropy_sampling max_disagreement_sampling, consensus_entropy_sampling
    )

    # =============================================================Query=====================================================
    idx_list = [] # 要query的sample list
    labeled_indices = deepcopy(n_idx) # 已经标注的sample list
    # n_queries = 8
    for i in range(n_queries):
        cur_kmeans = KMeans(n_initial+i+1, random_state=42)
        cur_kmeans.fit(X)

        # 获取初始质心
        cur_centroids = cur_kmeans.cluster_centers_
        # 为每个数据点分配聚类标签
        cur_labels = cur_kmeans.labels_
        labeled_indices = list(set(labeled_indices + idx_list))

        # We then identify the largest cluster that does not contain any labeled sample as the current most representative cluster
        largest_cluster, max_size = find_largest_unlabeled_cluster(cur_labels, labeled_indices)
        # largest_unlabeled_cluster下，所有样本的indices
        largest_cluster_samples_centroids = cur_centroids[largest_cluster]
        largest_cluster_samples_indices = [index for index, value in enumerate(cur_labels) if value == largest_cluster]

        # Basic: select the sample closest to its centroid for labeling
        if query_method == "basic":
            largest_cluster_samples = X[largest_cluster_samples_indices]
            nearest_local_index = findNearestSampleIndex(largest_cluster_samples, largest_cluster_samples_centroids, measurement)
            # 获取对应的全局索引
            nearest_sample_index = largest_cluster_samples_indices[nearest_local_index]
            committee.teach(
                X=X[[nearest_sample_index]],
                y=targets[[nearest_sample_index]]
            )
            idx_list.append(nearest_sample_index)

        # QBC: Use QBC to select a sample from the cluster for labeling
        elif query_method == "qbc":
            new_pool = X[largest_cluster_samples_indices]
            query_idx, query_instance = committee.query(new_pool)
            gloabl_idx = findIndex(X, query_instance[0])
            committee.teach(
                X=X[[gloabl_idx]],
                y=targets[[gloabl_idx]]
            )
            idx_list.append(gloabl_idx)

    return idx_list, committee

def rd_basic(train_data, n_initial, n_queries, measurement):
    # ==========================================================初始化数据==================================================
    train_data = np.array(train_data)
    X = train_data.reshape(train_data.shape[0],-1)
    # ==========================================================初始化learner==================================================
    n_idx, _ = find_init_centroids(X, n_initial, measurement)
    # =============================================================Query=====================================================
    idx_list = [] # 要query的sample list
    labeled_indices = deepcopy(n_idx) # 已经标注的sample list
    for i in range(n_queries-n_initial):
        cur_kmeans = KMeans(n_initial+i+1, random_state=42)
        cur_kmeans.fit(X)

        # 获取初始质心
        cur_centroids = cur_kmeans.cluster_centers_
        # 为每个数据点分配聚类标签
        cur_labels = cur_kmeans.labels_
        labeled_indices = list(set(labeled_indices + idx_list))

        # We then identify the largest cluster that does not contain any labeled sample as the current most representative cluster
        largest_cluster, max_size = find_largest_unlabeled_cluster(cur_labels, labeled_indices)
        # largest_unlabeled_cluster下，所有样本的indices
        largest_cluster_samples_centroids = cur_centroids[largest_cluster]
        largest_cluster_samples_indices = [index for index, value in enumerate(cur_labels) if value == largest_cluster]

        # Basic: select the sample closest to its centroid for labeling
        largest_cluster_samples = X[largest_cluster_samples_indices]
        nearest_local_index = findNearestSampleIndex(largest_cluster_samples, largest_cluster_samples_centroids, measurement)
        # 获取对应的全局索引
        nearest_sample_index = largest_cluster_samples_indices[nearest_local_index]
        idx_list.append(nearest_sample_index)

    return n_idx + idx_list # 最终的返回包括最开始找到的质心

# def gs(train_data, train_labels, n_initial, n_queries, measurement, init_indices):
#     # ==========================================================初始化数据==================================================
#     train_data = np.array(train_data)
#     train_labels = np.array(train_labels)
#     X = train_data.reshape(train_data.shape[0],-1)
#     lab_map = {
#         'left_hand': 1,
#         'right_hand':2
#     }
#     targets = np.array([lab_map[lab] for lab in train_labels])
#     # ==========================================================初始化选择的点==================================================

#     if init_indices:
#         n_idx = init_indices
#     else:
#         n_idx, _ = find_init_centroids(X, n_initial, measurement)

#     # =============================================================Query=====================================================
#     idx_list = [] # 要query的sample list
#     labeled_indices = deepcopy(n_idx) # 已经标注的sample list

#     for i in range(n_queries):


#     return idx_list
# =============================================================测试=====================================================
def compare_test(train_data, test_data, train_labels, test_labels, n_times, measurement, estimator, init_idx):
    # 导入测试数据
    lab_map = {
        'left_hand': 1,
        'right_hand':2
    }
    Y = test_data.reshape(test_data.shape[0],-1)
    targets_y = np.array([lab_map[lab] for lab in test_labels])

    rd_qbc_score_list = []
    rd_basic_score_list = []
    enhanced_qbc_score_list = []
    qbc_score_list = []
    rand_score_list = []

    for i in tqdm(range(n_times)):
        _, rd_qbc_committe = rd("qbc", train_data, train_labels, n_members=5, n_initial=12, n_queries=8, measurement=measurement, estimator=estimator)
        rd_qbc_score = rd_qbc_committe.score(Y, targets_y)
        rd_qbc_score_list.append(rd_qbc_score)

        _, rd_basic_committe = rd("basic", train_data, train_labels, n_members=5, n_initial=12, n_queries=8, measurement=measurement, estimator=estimator)
        rd_basic_score = rd_basic_committe.score(Y, targets_y)
        rd_basic_score_list.append(rd_basic_score)

        _, enhanced_qbc_committe = qbc(train_data, train_labels, n_members=5, n_initial=12, n_queries=8, estimator=estimator, init_idx=init_idx)
        enhanced_qbc_score = enhanced_qbc_committe.score(Y, targets_y)
        enhanced_qbc_score_list.append(enhanced_qbc_score)

        _, qbc_committe = qbc(train_data, train_labels, n_members=5, n_initial=12, n_queries=8, estimator=estimator)
        qbc_score = qbc_committe.score(Y, targets_y)
        qbc_score_list.append(qbc_score)

        _, rand_committe = rand_select(train_data, train_labels, n_members=5, n_initial=20, estimator=estimator)
        rand_score = rand_committe.score(Y, targets_y)
        rand_score_list.append(rand_score)

        rd_qbc_mean = np.mean(rd_qbc_score_list)
        rd_basic_mean = np.mean(rd_basic_score_list)
        enhanced_qbc_mean = np.mean(enhanced_qbc_score_list)
        qbc_mean = np.mean(qbc_score_list)
        random_mean = np.mean(rand_score_list)

        print(measurement)
        print(rd_qbc_mean, rd_basic_mean, enhanced_qbc_mean, qbc_mean, random_mean)
    return rd_qbc_mean, rd_basic_mean, enhanced_qbc_mean, qbc_mean, random_mean

if __name__=='__main__':
    # ==========================================================载入数据==================================================
    dataset_name = "2a"
    sub_index = 1
    test_id = 1
    repeat_times = 1
    measurement = "cs" # "ed"欧氏距离；"cs"余弦相似度
    estimator = RandomForestClassifier
    para_dict = {
        '2a': 'mi',
        '2b': 'mi',
        'BNCI2015_001': 'mi',
        'BNCI2014_002': 'mi',
        'Kalunga2016': 'ssvep',
    }

    for i in range(1, 2):
        train_data, test_data, train_labels, test_labels = get_moabb_data(dataset_name, i, test_id, para_dict[dataset_name])

        lab_map = {
            'left_hand': 1,
            'right_hand':2
        }
        train_data = np.array(train_data)
        X = train_data.reshape(train_data.shape[0],-1)
        targets = np.array([lab_map[lab] for lab in train_labels])

        test_data = np.array(test_data)
        Y = test_data.reshape(test_data.shape[0],-1)
        targets_y = np.array([lab_map[lab] for lab in test_labels])

        init_id_list, _ = find_init_centroids(X, init_num=12, measurement=measurement)
        rd_qbc_mean, rd_basic_mean, enhanced_qbc_mean, qbc_mean, random_mean = compare_test(train_data, test_data, train_labels, test_labels, repeat_times, measurement, estimator, init_idx=init_id_list)
        # rd_qbc_mean, rd_basic_mean, enhanced_qbc_mean, qbc_mean, random_mean = compare_test(train_data, test_data, train_labels, test_labels, repeat_times, "cs", init_idx=init_id_list)


        # with open(f"log/active_learn_log", 'a',  encoding='utf-8') as file:
        #     file.write(f"重复{repeat_times}次 ")
        #     file.write(f"--- 被试{i} ---\n")
        #     file.write(f"{rd_qbc_mean} ")
        #     file.write(f"{rd_basic_mean} ")
        #     file.write(f"{enhanced_qbc_mean} ")
        #     file.write(f"{qbc_mean} ")
        #     file.write(f"{random_mean} ")
        #     file.write("\n")  # 添加换行，便于区分记录


# =============================================================画图=====================================================

    # # tsne降维
    # tsne = TSNE(n_components=2, random_state=42)
    # X_tsne = tsne.fit_transform(X)

    # label_to_color={
    #     1: 'grey',
    #     2: 'yellow'
    # }

    # plt.figure(figsize=(2 * 7, 7))
    # plt.subplot(1, 2, 1)
    # plt.scatter(x=X_tsne[:,0], y=X_tsne[:,1], c=[label_to_color[tar] for tar in targets], cmap='viridis', s=50)
    # plt.scatter(x=X_tsne[init_id_list, 0], y=X_tsne[init_id_list, 1], c='red', s=20, label='Highlighted')
    # plt.show()

    # plt.figure(figsize=((n_members+1) * 7, 7))
    # plt.subplot(1, n_members+1, 1)
    # plt.scatter(x=X_tsne[:,0], y=X_tsne[:,1], c=[label_to_color[tar] for tar in targets], cmap='viridis', s=50)
    # plt.scatter(x=X_tsne[idx_list, 0], y=X_tsne[idx_list, 1], c='red', s=20, label='Highlighted')
    # for idx in idx_list:
    #     plt.text(X_tsne[idx, 0], X_tsne[idx, 1], f'Index {idx}', fontsize=10, ha='right', color='red')
    # plt.title('ground truth label and qbc queried samples')

    # for learner_idx, learner in enumerate(committee):
    #     plt.subplot(1, n_members+1, learner_idx + 2)
    #     plt.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], c=[label_to_color[tar] for tar in learner.predict(X)], cmap='viridis', s=50)
    #     plt.scatter(x=X_tsne[idx_list, 0], y=X_tsne[idx_list, 1], c='red', s=20, label='Highlighted')
    #     for idx in idx_list:
    #         plt.text(X_tsne[idx, 0], X_tsne[idx, 1], f'Index {idx}', fontsize=10, ha='right', color='red')
    #     plt.title('Learner no. %d initial predictions' % (learner_idx + 1))

    # plt.show()
