# -*- coding: utf-8 -*-
# @bref :使用tensorflow做中文情感分析
import numpy as np
import tensorflow as tf
import random
import pickle
from collections import Counter
import os

real_dir_path = os.path.split(os.path.realpath(__file__))[0]
pos_file = os.path.join(real_dir_path, 'data/pos.txt')
neg_file = os.path.join(real_dir_path, 'data/neg.txt')

from pyltp import Segmentor
seg = Segmentor()
seg.load('/root/git/ltp_data/cws.model')

def create_vocab(pos_file, neg_file):
    vocab = []
    def process_file(file):
        with open(file, 'r') as f:
            v = []
            lines = f.readlines()
            for line in lines:
                words = seg.

