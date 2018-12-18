#!/usr/bin/python
# -*- encoding:utf-8 -*-

"""
@author : kelvin
@file : kNN
@time : 2017/3/24 14:54
@description :

"""
from numpy import *
import operator


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]            # 取得第一维度的大小，这里是训练样本的大小
    # 欧氏距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet    # tile（A，res）重复构建A构建数组，res给出次数，计算到每个训练样本的距离
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)    # 矩阵每行里的元素相加
    distance = sqDistance**0.5
    sortedDistIndicies = distance.argsort()      # 按小到大排序，返回原来索引列表
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        # 如果voteIlabel在classCount字典中，则变量加1，如果没有，则新建置1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1      # D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None.
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)   # 直接用sorted(classCount, reverse=True)也可以
    return sortedClassCount[0][0]

# 执行算法
# group, labels = createDataSet()
# print classify0([0, 0], group, labels, 3)

# 使用KNN改进约会对象的配对效果
# 每个样本三个特征：每年获得的飞行常客里程数， 玩视频游戏所耗的时间比， 每周消费的冰淇凌公升数
# 将文本数据转换为numpy
def file2matrix(filename):
    fr = open(filename)
    array_lines = fr.readlines()
    number_of_lines = len(array_lines)
    return_mat = zeros((number_of_lines, 3))   # 创建返回的NumPy矩阵
    class_label_vector = []
    index = 0
    for line in array_lines:
        line = line.strip()
        list_from_line = line.split('\t')    # \t表示键盘上的tab键
        return_mat[index, :] = list_from_line[0:3]   # 数据填进特征矩阵
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector

# dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')

# 可视化数据
# import matplotlib
# import matplotlib.pyplot as plt
#
# fig = plt.figure()      # Creates a new figure
# ax = fig.add_subplot(111)    # # equivalent but more general fig.add_subplot(1,1,1)
# # ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2])     # 绘制散点图
# ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1], 15.0*array(dating_labels), 20.0*array(dating_labels))
# plt.show()

# 归一化数据
def auto_norm(dataset):
    min_vals = dataset.min(0)       # 取每列的最小值，numpy中的min参数0表示取每列的最小，1表示取每行的最小
    max_vals = dataset.max(0)
    ranges = max_vals - min_vals    # 每列数值的取值范围
    norm_dataset = zeros(shape(dataset))
    m = dataset.shape[0]            # 取得行数
    norm_dataset = dataset - tile(min_vals, (m, 1))    # tile()将min_vals复制成输入矩阵相同大小的矩阵 m-->要复制的行数
    norm_dataset = norm_dataset/tile(ranges, (m, 1))   # 具体特征值相除
    return norm_dataset, ranges, min_vals

# norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
# print norm_mat[0:10]
# print ranges
# print min_vals

# 测试算法的错误率
def dating_class_test():
    ho_ratio = 0.30     # 测试集的比率
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m*ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):   # 前10%作为测试集
        classifier_result = classify0(norm_mat[i, :], norm_mat[num_test_vecs:m, :], dating_labels[num_test_vecs:m], 6)
        print 'the classifier came back with: %d, the real answer is: %d' % (classifier_result, dating_labels[i])
        if classifier_result != dating_labels[i]:
            error_count += 1.0
    print 'the total error rate is : %f' % (error_count/float(num_test_vecs))

# dating_class_test()

# 构建完整可用系统
def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    precent_tats = float(raw_input('percentage of time spent on playing video games?'))
    ff_miles = float(raw_input('frequent flier miles earned per year?'))
    ice_cream = float(raw_input('liters of ice cream consumed per year?'))
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    in_arr = array([ff_miles, precent_tats, ice_cream])
    classifier_result = classify0((in_arr-min_vals)/ranges, norm_mat, dating_labels, 6)
    print "you will probably like this person: ", result_list[classifier_result-1]

# classify_person()



