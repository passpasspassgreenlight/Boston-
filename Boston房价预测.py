from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
b_datas = datasets.load_boston()  # 加载数据集
x = b_datas.data  # 获取特征数据
y = b_datas.target  # 获取标签数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)  # 划分训练集和测试集
lr = LinearRegression()  # 创建线性回归模型
lr.fit(x_train, y_train)  # 拟合数据学习模型参数
y_test_predict = lr.predict(x_test)  # 预测测试数据结果
# print(y_test_predict)
# print(y_test)
error_1 = mean_squared_error(y_test, y_test_predict)  # 测试误差
print("测试数据的误差：", error_1)
y_train_predict = lr.predict(x_train)
error_2 = mean_squared_error(y_train, y_train_predict)  # 训练误差
print("训练数据的误差：", error_2)

from sklearn import datasets
import math
import matplotlib.pyplot as plt
house = datasets.load_boston()
x = house.data
y = house.target
nums = len(house.feature_names)
columns = 3
rows = math.ceil(nums / columns)
plt.figure(figsize=(10, 12))
for i in range(nums):
    plt.subplot(rows, columns, i + 1)
    plt.plot(x[:, i], y, "b+")
    plt.title(house.feature_names[i])
plt.subplots_adjust(hspace=0.8)
plt.show()

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
house = datasets.load_boston()
x = house.data
y = house.target
stand = StandardScaler()
stand_x = stand.fit_transform(x)
best = SelectKBest(f_regression, k=3)
best_x = best.fit_transform(stand_x, y)

train_x, test_x, train_y, test_y = train_test_split(best_x, y, test_size=0.2)
lr = LinearRegression()
lr.fit(train_x, train_y)
predict_y = lr.predict(test_x)
error = mean_squared_error(test_y, predict_y)
print("误差为：", error)

plt.plot(predict_y, "r-", label="Predict_value")
plt.plot(test_y, "b-", label="True_value")
plt.legend()
plt.show()

# print(best_x)
# best_index = best.get_support()
# print(stand_x[:, best_index])

