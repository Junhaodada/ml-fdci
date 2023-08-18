"""使用numpy实现kmeans聚类"""
import numpy as np
from data import X


def euclidean_distance(x, y):
    """计算欧氏距离
    Args:
        x: 向量x
        y: 向量y

    Returns:
        x和y的欧氏距离
    """
    distance = 0
    for i in range(len(x)):
        distance += (x[i] - y[i]) ** 2
    return np.sqrt(distance)


def centroids_init(X: np.ndarray, k: int):
    """质心初始化

    Args:
        X: 训练样本
        k: 质心数，即聚类数

    Returns:
        质心矩阵
    """
    m, n = X.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        centroid = X[np.random.choice(range(m))]
        centroids[i] = centroid
    return centroids


def closest_centroid(x, centroids):
    """定义样本所属最近质心的索引

    Args:
        x: 单个样本
        centroids: 质心矩阵

    Returns:
        距离质心最近的索引
    """
    closest_i, closed_dist = 0, float('inf')
    for i, centroid in enumerate(centroids):
        distance = euclidean_distance(x, centroid)
        if distance < closed_dist:
            closed_dist = distance
            closest_i = i
    return closest_i


def build_clusters(centroids, k, X):
    """分配样本与构建簇

    Args:
        centroids: 质心矩阵
        k: 质心数
        X: 训练样本

    Returns:
        聚类簇
    """
    clusters = [[] for _ in range(k)]
    for x_i, x in enumerate(X):
        centroid_i = closest_centroid(x, centroids)
        clusters[centroid_i].append(x_i)
    return clusters


def calculate_centroids(clusters, k, X):
    """更新质心

    Args:
        clusters: 当前聚类簇
        k: 质心数
        X: 训练样本

    Returns:
        更新后的聚类簇
    """
    n = X.shape[1]
    centroids = np.zeros((k, n))
    for i, cluster in enumerate(clusters):
        centroid = np.mean(X[cluster], axis=0)
        centroids[i] = centroid
    return centroids


def get_cluster_labels(clusters, X):
    """获取每个样本所属的聚类类别

    Args:
        clusters: 当前聚类簇
        X: 训练样本

    Returns:
        样本对应的聚类类别
    """
    y_pred = np.zeros(X.shape[0])
    for cluster_i, cluster in enumerate(clusters):
        for sample_i in cluster:
            y_pred[sample_i] = cluster_i
    return y_pred


def kmeans(X, k, max_iter):
    """kmeans算法流程

    Args:
        X: 训练样本
        k: 质心个数
        max_iter: 最大迭代次数

    Returns:
        样本对应聚类的类别
    """
    # 初始化质心
    centroids = centroids_init(X, k)
    clusters = [[] for _ in range(k)]
    for _ in range(max_iter):
        # 根据当前质心进行聚类
        clusters = build_clusters(centroids, k, X)
        # 保存当前质心
        cur_centroids = centroids
        # 根据当前聚类结果计算新的质心
        centroids = calculate_centroids(clusters, k, X)
        # 判断是否收敛
        diff = centroids - cur_centroids
        if not diff.any():
            break
    return get_cluster_labels(clusters, X)


if __name__ == '__main__':
    labels = kmeans(X, k=2, max_iter=10)
    print(labels)
