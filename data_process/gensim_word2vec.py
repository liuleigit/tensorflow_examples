# -*- coding: utf-8 -*-
# @Time    : 17/5/22 下午11:06
# @Author  : liulei
# @Brief   : 使用gensim 获得word2vec
# @File    : gensim_word2vec.py
# @Software: PyCharm Community Edition
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
#df = pd.read_csv('./data/data_cut/data_cut.csv')
#df = pd.read_csv('./data/data_cut/test')
##sentences = df['doc']
#print type(sentences)
#print sentences.tolist()[0]
#sentences = [i.decode('utf-8') for i in sentences.tolist()]

model = Word2Vec(LineSentence('./data/data_cut/test'), size=300, window=5, min_count=1, workers=2)
model.save('./word2vec.model')
print model.vocab
print type(model.vocab)
for i in model.vocab.keys():
    print type(i)
    print i
#model = Word2Vec.load('word2vec_model')
#print model.wv[u'篮球']
'''
print model.similarity(u"刘国梁", u"张继科")
print model.wv[u'张继科']
print model.wv['张继科']
print model.wv[u'篮板']
print model.wv['篮板']
'''
