# -*- coding: utf-8 -*-
# @Time    : 17/5/22 下午11:06
# @Author  : liulei
# @Brief   : 使用gensim 获得word2vec
# @File    : gensim_word2vec.py
# @Software: PyCharm Community Edition
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

df = pd.read_csv('./data/data_cut/data_cut.csv')
#df = pd.read_csv('./data/data_cut/test')
sentences = df['doc']
line_sent = []
for s in sentences:
    line_sent.append(s.decode('utf-8').split())

print 'begin to train. size of sentencts is {} '.format(len(line_sent))
model = Word2Vec(line_sent, size=300, window=5, min_count=5, workers=10)
print 'finish to train word2vec!'
model.save('./word2vec.model')
print model.vector_size
i = 0
for v in model.vocab.keys():
    i += 1
    print v
print model.vocab[u'欧冠']
