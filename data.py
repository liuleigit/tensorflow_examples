# -*- coding: utf-8 -*-
# @bref : 提供数据
import os
real_dir_path = os.path.split(os.path.realpath(__file__))[0]
data_path = os.path.join(real_dir_path, 'NewsFileCut')
chanls = os.listdir(data_path)
for ch in chanls:
    print ch
    save_path = os.path.join(real_dir_path, 'data', ch)
    f = open(save_path, 'w')
    n = 0
    ch_path = os.path.join(data_path, ch)
    files = os.listdir(ch_path)
    for ff in files:
        fr = open(os.path.join(ch_path, ff), 'r')
        f.write(' '.join(fr.read().split()) + '\n')

    f.close()



