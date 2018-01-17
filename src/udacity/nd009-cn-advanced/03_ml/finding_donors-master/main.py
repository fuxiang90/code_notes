#!coding=utf-8
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # 允许为DataFrame使用display()

# 导入附加的可视化代码visuals.py
import visuals as vs


# 导入人口普查数据
data = pd.read_csv("census.csv")

# 成功 - 显示第一条记录
display(data.head(n=1))

#1
# # TODO：总的记录数
# n_records = len(data)
#
# # TODO：被调查者的收入大于$50,000的人数
# n_greater_50k = 0
# for index, each in data.iterrows():
#     if each['income'] == ">50K":
#         n_greater_50k += 1
#
#
# # TODO：被调查者的收入最多为$50,000的人数
#
# n_at_most_50k = 0
# for index, each in data.iterrows():
#     if each['income'] == "<=50K":
#         n_at_most_50k += 1
#
# # TODO：被调查者收入大于$50,000所占的比例
# greater_percent = n_greater_50k*1.0/n_records
#
# # 打印结果
# print "Total number of records: {}".format(n_records)
# print "Individuals making more than $50,000: {}".format(n_greater_50k)
# print "Individuals making at most $50,000: {}".format(n_at_most_50k)
# print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)


#2
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

from sklearn.preprocessing import MinMaxScaler

# 初始化一个 scaler，并将它施加到特征上
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# 显示一个经过缩放的样例记录
display(features_raw.head(n = 1))


# TODO：使用pandas.get_dummies()对'features_raw'数据进行独热编码
import pandas as pd
features = pd.get_dummies(features_raw)

# TODO：将'income_raw'编码成数字值
income = income_raw.replace(['>50K', '<=50K'], [1, 0])

# 打印经过独热编码之后的特征数量
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))
display(features.head(n = 1))
display(income.head(n = 1))
print len(income.index)
# 移除下面一行的注释以观察编码的特征名字
#print encoded

#3
# 导入 train_test_split
from sklearn.model_selection import train_test_split

# 将'features'和'income'数据切分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0,
                                                    stratify = income)
# # 将'X_train'和'y_train'进一步切分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,
                                                    stratify = y_train)

# # 显示切分的结果
print "Training set has {} samples.".format(X_train.shape[0])
print "Validation set has {} samples.".format(X_val.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


# 4
#不能使用scikit-learn，你需要根据公式自己实现相关计算。

# TODO： 计算准确率
accuracy_cnt = 0
for value in y_val:
    if value > 0:
        accuracy_cnt += 1
accuracy = accuracy_cnt*1.0/len(y_val)


# TODO： 计算查准率 Precision
precision = accuracy_cnt*1.0/len(y_val)

# TODO： 计算查全率 Recall
recall = 1.0

# TODO： 使用上面的公式，设置beta=0.5，计算F-score
fscore = (1 +0.5*0.5)* (precision*recall/(0.5*0.5*precision+ recall))

# 打印结果
print "Naive Predictor on validation data: \n \
    Accuracy score: {:.4f} \n \
    Precision: {:.4f} \n \
    Recall: {:.4f} \n \
    F-score: {:.4f}".format(accuracy, precision, recall, fscore)


# TODO：从sklearn中导入两个评价指标 - fbeta_score和accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score


def train_predict(learner, sample_size, X_train, y_train, X_val, y_val):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_val: features validation set
       - y_val: income validation set
    '''

    results = {}

    # TODO：使用sample_size大小的训练数据来拟合学习器
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time()  # 获得程序开始时间
    ml = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()  # 获得程序结束时间

    # TODO：计算训练时间
    results['train_time'] = end - start
    print  end - start
    # TODO: 得到在验证集上的预测值
    #       然后得到对前300个训练数据的预测结果
    start = time()  # 获得程序开始时间
    predictions_val = ml.predict(X_val)
    predictions_train = ml.predict(X_train[:300])
    end = time()  # 获得程序结束时间

    # TODO：计算预测用时
    results['pred_time'] = end - start

    # TODO：计算在最前面的300个训练数据的准确率
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    # TODO：计算在验证上的准确率
    results['acc_val'] = accuracy_score(y_val, predictions_val)

    # TODO：计算在最前面300个训练数据上的F-score
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)

    # TODO：计算验证集上的F-score
    results['f_val'] = fbeta_score(y_val, predictions_val ,beta=0.5)

    # 成功
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)

    # 返回结果
    return results



# TODO：从sklearn中导入三个监督学习模型

from sklearn import linear_model
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier

# TODO：初始化三个模型
clf_A = linear_model.LogisticRegression()
clf_B = tree.DecisionTreeClassifier()
clf_C = GradientBoostingClassifier()

# TODO：计算1%， 10%， 100%的训练数据分别对应多少点
samples_1 = int(0.01*len(X_train))
samples_10 = int(0.1*len(X_train))
samples_100 = 1*len(X_train)

print samples_1,samples_10,samples_100
# 收集学习器的结果
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_val, y_val)

# 对选择的三个模型得到的评价结果进行可视化
vs.evaluate(results, accuracy, fscore)


print "end "


# TODO：导入'GridSearchCV', 'make_scorer'和其他一些需要的库

# TODO：初始化分类器
clf = None

# TODO：创建你希望调节的参数列表
parameters = None

# TODO：创建一个fbeta_score打分对象
scorer = None

# TODO：在分类器上使用网格搜索，使用'scorer'作为评价函数
grid_obj = None

# TODO：用训练数据拟合网格搜索对象并找到最佳参数

# 得到estimator
best_clf = grid_obj.best_estimator_

# 使用没有调优的模型做预测
predictions = (clf.fit(X_train, y_train)).predict(X_val)
best_predictions = best_clf.predict(X_val)

# 汇报调参前和调参后的分数
print "Unoptimized model\n------"
print "Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions))
print "F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions))
print "Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5))