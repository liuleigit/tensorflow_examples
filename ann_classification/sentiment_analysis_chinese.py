# -*- coding: utf-8 -*-
# @bref :使用tensorflow做中文情感分析
import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import os
import traceback

real_dir_path = os.path.split(os.path.realpath(__file__))[0]
pos_file = os.path.join(real_dir_path, 'data/pos.txt')
neg_file = os.path.join(real_dir_path, 'data/neg.txt')

#使用哈工大分词和词性标注
from pyltp import Segmentor, Postagger
seg = Segmentor()
seg.load('/root/git/ltp_data/cws.model')
poser = Postagger()
poser.load('/root/git/ltp_data/pos.model')
real_dir_path = os.path.split(os.path.realpath(__file__))[0] #文件所在路径
stop_words_file = os.path.join(real_dir_path, '../util/stopwords.txt')
#定义允许的词性
allow_pos_ltp = ('a', 'i', 'j', 'n', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'v', 'ws')
def create_vocab(pos_file, neg_file):
    def process_file(file_path):
        with open(file_path, 'r') as f:
            v = []
            lines = f.readlines()
            for line in lines:
                words = seg.segment(''.join(line.split()))
                poses = poser.postag(words)
                stopwords = {}.fromkeys([line.rstrip() for line in open(stop_words_file)])
                sentence = []
                for i, pos in enumerate(poses):
                    if (pos in allow_pos_ltp) and (words[i] not in stopwords):
                        sentence.append(words[i])
                v.append(' '.join(sentence))
            return v
    sent = process_file(pos_file)
    #sent += process_file(neg_file)
    tf_v = CountVectorizer(max_df=0.9, min_df=1)
    tf = tf_v.fit_transform(sent)
    print tf_v.vocabulary_
    return tf_v.vocabulary_



if __name__ == '__main__':
    try:
        create_vocab(pos_file, neg_file)
    except:
        traceback.print_exc()