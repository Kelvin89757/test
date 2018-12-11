#!/usr/bin/python
# -*- encoding:utf-8 -*-

"""
@author : kelvin
@file : kNN_digits
@time : 2017/3/25 20:07
@description : 

"""
from kNN import classify0
from numpy import *
from os import listdir
import time


def img2vector(filename):
    """
    将32x32的图像转化为1x1024的向量
    :param filename:
    :return:
    """
    return_vect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):              # 用for line in readlines()也可以
        line_str = fr.readline()      # 一行一行的读
        for j in range(32):
            return_vect[0, 32*i+j] = int(line_str[j])
    fr.close()
    return return_vect

# test = img2vector(r'testDigits\0_0.txt')
# print test[0, 0:31]

# 手写数字识别系统测试代码
def handwriting_class():
    hw_labels = []
    training_file_list = listdir('trainingDigits')    # 获取目录下的文件名
    m = len(training_file_list)
    training_mat = zeros((m, 1024))
    for i in range(m):
        file_name_str = training_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num = int(file_str.split('_')[0])
        hw_labels.append(class_num)
        training_mat[i, :] = img2vector(r'trainingDigits/%s' % file_name_str)
    test_file_list = listdir('testDigits')
    error_count = 0.0
    m_test = len(test_file_list)
    for i in range(m_test):
        file_name_str = test_file_list[i]
        file_str = file_name_str.split('.')[0]
        class_num = int(file_str.split('_')[0])
        vector_under_test = img2vector(r'testDigits/%s' % file_name_str)
        classifier_result = classify0(vector_under_test, training_mat, hw_labels, 5)
        print 'the classifier came back with: %d, the real answer is: %d' % (classifier_result, class_num)
        if classifier_result != class_num:
            error_count += 1.0
    print '\nthe total number of errors is : %d' % error_count
    print '\nthe total error rate is : %f' % (error_count/float(m_test))

start = time.clock()
handwriting_class()
print "time used: ", time.clock()-start

