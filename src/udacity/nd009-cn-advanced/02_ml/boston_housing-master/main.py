#!coding=utf-8
import numpy as np
import pandas as pd
import visuals as vs # Supplementary code
import sklearn

# 检查你的Python版本
from sys import version_info
if version_info.major != 2 and version_info.minor != 7:
    raise Exception('请使用Python 2.7来完成此项目')

# 载入波士顿房屋的数据集
data = pd.read_csv('bj_housing.csv')
prices = data['Value']
features = data.drop('Value', axis=1)

# 完成
print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)



#TODO 1

#目标：计算价值的最小值

minimum_price = np.min(prices)

#目标：计算价值的最大值
maximum_price = np.max(prices)

#目标：计算价值的平均值
mean_price = np.average(prices)

#目标：计算价值的中值
median_price = np.median(prices)

#目标：计算价值的标准差
std_price = np.std(prices)

#目标：输出计算的结果
print "Statistics for Boston housing dataset:\n"
print "Minimum price: ${:,.2f}".format(minimum_price)
print "Maximum price: ${:,.2f}".format(maximum_price)
print "Mean price: ${:,.2f}".format(mean_price)
print "Median price ${:,.2f}".format(median_price)
print "Standard deviation of prices: ${:,.2f}".format(std_price)

# TODO 2

# 提示： 导入train_test_split

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features, prices, test_size=0.3)


def performance_metric(y_true, y_predict):
    """计算并返回预测值相比于预测值的分数"""

    score = sklearn.metrics.r2_score(y_true, y_predict)

    return score


def fit_model(X, y):
    """ 基于输入数据 [X,y]，利于网格搜索找到最优的决策树模型"""


    cross_validator = sklearn.model_selection.KFold(30, shuffle=True)

    regressor = sklearn.tree.DecisionTreeRegressor()

    params = {"max_depth":[1,2,3,4,5,6,7,8,9,10]}

    scoring_fnc = sklearn.metrics.make_scorer(performance_metric)

    grid = sklearn.model_selection.GridSearchCV(regressor, params, scoring_fnc,cv=cross_validator)

    # 基于输入数据 [X,y]，进行网格搜索
    grid = grid.fit(X, y)

    # 返回网格搜索后的最优模型
    return grid.best_estimator_


# 基于训练数据，获得最优模型
optimal_reg = fit_model(X_train, y_train)

# 输出最优模型的 'max_depth' 参数
print "Parameter 'max_depth' is {} for the optimal model.".format(optimal_reg.get_params()['max_depth'])


# 提示：你可能需要用到 X_test, y_test, optimal_reg, performance_metric
# 提示：你可能需要参考问题10的代码进行预测
# 提示：你可能需要参考问题3的代码来计算R^2的值
pred_y = optimal_reg.predict(X_test)
r2 = performance_metric(y_test,pred_y)

print "Optimal model has R^2 score {:,.2f} on test data".format(r2)