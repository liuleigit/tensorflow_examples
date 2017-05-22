# -*- coding: utf-8 -*-
# @Time    : 17/5/22 下午11:06
# @Author  : liulei
# @Brief   : 使用gensim 获得word2vec
# @File    : gensim_word2vec.py
# @Software: PyCharm Community Edition
import gensim
import pandas as pd
from gensim.models import Word2Vec

df = pd.read_csv('./data/data_cut/data_cut.csv')
sentences = df['doc']

model = Word2Vec(sentences, size=300, window=5, min_count=10, workers=10)
model.save('./word2vec_model')
print model.wv['张继科']
