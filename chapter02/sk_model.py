"""基于sklearn的线性回归模型"""

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from data import *

# 定义线性回归模型
regr = linear_model.LinearRegression()
# 训练
regr.fit(X_train, y_train)
# 使用模型预测测试数据
y_pred = regr.predict(X_test)  # (89, 1)
# 模型效果
print('mean square error:', mean_squared_error(y_test, y_pred))
print('r square score:', r2_score(y_test, y_pred))
