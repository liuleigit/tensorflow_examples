# -*- coding: utf-8 -*-
# @Time    : 17/5/23 下午4:51
# @Author  : liulei
# @Brief   : 为cnn准备数据
# @File    : data_process.py
# @Software: PyCharm Community Edition
from gensim.models import Word2Vec
import pandas as pd


def load_word2vec(fname):
    model = Word2Vec.load(fname)
    return model.wv

def load_data_and_label(fname):
    '''
    读取csv文件, 去除文本和label
    '''
    df = pd.read_csv(fname)
    df = df.dropna()


