"""线性回归代码数据集"""

# 导入数据集
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle

# 创建数据集
diabetes = load_diabetes()
data, target = diabetes.data, diabetes.target
# print(data.shape) # (442, 10)
# print(target.shape) # (442,)

# 打乱数据集的顺序
X, y = shuffle(data, target, random_state=12)
# print(X[:5])
# print(y[:5])

# 8:2划分训练集和测试集
offset = int(X.shape[0] * 0.8)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# 将y转为列向量形式
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

# 输出数据集的shape
print("X_train's shape:", X_train.shape)
print("y_train's shape:", y_train.shape)
print("X_test's shape:", X_test.shape)
print("y_test's shape:", y_test.shape)
