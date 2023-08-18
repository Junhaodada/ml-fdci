from sklearn.cluster import KMeans
from data import X

kmeans = KMeans(n_clusters=2, n_init='auto', random_state=0).fit(X)
print(kmeans.__str__())
print(kmeans.labels_)
