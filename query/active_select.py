import numpy as np # type: ignore
from scipy.spatial.distance import cosine # type: ignore

class KMeans:
    def __init__(self, num_clusters, max_iter=300, tol=1e-4):
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X):
        # 1. 随机选择 k 个初始质心
        centroid_indices = np.random.choice(X.shape[0], self.num_clusters, replace=False)
        centroids = X[centroid_indices]

        # 2. 迭代
        for _ in range(self.max_iter):
            # 计算距离矩阵：扩展 X 以计算每个样本与质心的距离
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            # 聚类
            clusters = np.argmin(distances, axis=1)

            # 计算聚类中所有点的平均值为质心
            new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(self.num_clusters)])
            
            # 比较新旧质心的距离是否小于容差，如果是则收敛
            if np.all(np.abs(new_centroids - centroids) < self.tol):
                centroids = new_centroids
                break

            # 如果继续迭代，需要将centroids更新
            centroids = new_centroids

        self.__clusters = clusters
        self.__centroids = centroids

    # 返回所有样本所属的簇
    def get_clusters(self):
        return self.__clusters

    # 返回 k 个质心的坐标
    def get_centroids(self):
        return self.__centroids


# Placeholder functions for QBC, EMCM, and GS
def select_sample_qbc(cluster_data):
    # Implement QBC strategy here
    return cluster_data[0]

def select_sample_emcm(cluster_data):
    # Implement EMCM strategy here
    return cluster_data[0]

def select_sample_gs(cluster_data):
    # Implement GS strategy here
    return cluster_data[0]

class BasicRD:
    def __init__(self, init_num, final_num):
        self.init_num = init_num
        self.final_num = final_num

    def rd_iterate(self, X, selection_strategy="qbc"):
        # Step 1: 初始标记样本选择
        kier = KMeans(self.init_num)
        kier.fit(X)
        
        # 获取初始质心
        clusters = kier.get_clusters()
        centroids = kier.get_centroids()
        
        # 从每个聚类中选择距离质心最近的样本
        labeled_samples = set()
        for i in range(self.init_num):
            cluster_indices = np.where(clusters == i)[0]
            cluster_data = X[cluster_indices]
            distances = np.linalg.norm(cluster_data - centroids[i], axis=1)
            nearest_index_in_cluster = cluster_indices[np.argmin(distances)]
            labeled_samples.add(nearest_index_in_cluster)

        # Step 2: 从 m = d+1 到 M 的迭代阶段
        for m in range(self.init_num + 1, self.final_num + 1):
            kier = KMeans(m)
            kier.fit(X)
            clusters = kier.get_clusters()
            centroids = kier.get_centroids()

            # 找到未包含标记样本的最大簇
            unique_clusters, counts = np.unique(clusters, return_counts=True) # 返回去重后的ndarray
            largest_cluster = None
            for cluster_id in unique_clusters[np.argsort(-counts)]: # 默认从小到大，变成负数后，从大到小
                cluster_indices = np.where(clusters == cluster_id)[0]
                if not labeled_samples.intersection(cluster_indices):
                    largest_cluster = cluster_indices
                    break

            if largest_cluster is None:
                continue

            # 从该簇中根据策略选择样本
            cluster_data = X[largest_cluster]
            if selection_strategy == "qbc":
                selected_sample = select_sample_qbc(cluster_data)
            elif selection_strategy == "emcm":
                selected_sample = select_sample_emcm(cluster_data)
            elif selection_strategy == "gs":
                selected_sample = select_sample_gs(cluster_data)
            elif selection_strategy == "basic":
                # 默认选择距离质心最近的样本
                centroid = centroids[cluster_id]
                distances = np.linalg.norm(cluster_data - centroid, axis=1)
                selected_sample = cluster_data[np.argmin(distances)]

            # 标记选中的样本并存储其索引
            selected_index = np.where((X == selected_sample).all(axis=1))[0][0]
            labeled_samples.add(selected_index)

        return X[list(labeled_samples)], list(labeled_samples)

# 使用示例
# X = np.random.rand(57, 30)
# RD_instance = BasicRD(4, 8)
# selected_samples, selected_indices = RD_instance.rd_iterate(X, selection_strategy="basic")

# print(selected_samples, selected_indices)



def find_k_similar(k, target, group):
    cosine_dist_list = [cosine(target.reshape(-1), tri.reshape(-1)) for tri in group]
    selected_indices = np.argsort(cosine_dist_list)[1:k+1]
    return selected_indices

def find_k_similar_for_each_label(k, target, group, labels):
    cosine_dist_list = [cosine(target.reshape(-1), tri.reshape(-1)) for tri in group]
    cos_sim_acs_indices = np.argsort(cosine_dist_list)
    selected_labels = set()
    selected_indices = []
    count = 0
    for i in range(1, len(cos_sim_acs_indices)):
        label = labels[cos_sim_acs_indices[i]]
        if not label in selected_labels:
            selected_labels.add(label)
            selected_indices.append(cos_sim_acs_indices[i])
            count+=1
        if count == k:
            break

    return selected_indices





