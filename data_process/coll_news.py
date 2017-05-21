# -*- coding: utf-8 -*-
# @Time    : 17/5/21 下午10:02
# @Author  : liulei
# @Brief   : 
# @File    : coll_news.py
# @Software: PyCharm Community Edition
import os
import psycopg2
import time
import traceback
from bs4 import BeautifulSoup
import pandas as pd

real_dir_path = os.path.split(os.path.realpath(__file__))[0] #文件所在路径

POSTGRE_USER = 'postgres'
POSTGRE_PWD = 'ly@postgres&2015'
#POSTGRE_HOST = '120.27.163.25'
POSTGRE_HOST = '10.47.54.175'
POSTGRE_DBNAME = 'BDP'
#POSTGRES = "postgresql://postgres:ly@postgres&2015@120.27.163.25:5432/BDP"
POSTGRES = "postgresql://postgres:ly@postgres&2015@10.47.54.175:5432/BDP"
def get_postgredb():
    try:
        connection = psycopg2.connect(database=POSTGRE_DBNAME, user=POSTGRE_USER, password=POSTGRE_PWD, host=POSTGRE_HOST,)
        cursor = connection.cursor()
        return connection, cursor
    except:    #出现异常,再次连接
        try:
            time.sleep(2)
            connection = psycopg2.connect(database=POSTGRE_DBNAME, user=POSTGRE_USER, password=POSTGRE_PWD, host=POSTGRE_HOST,)
            cursor = connection.cursor()
            return connection, cursor
        except:
            traceback.print_exc()
            raise


def join_csv(in_files, out_file, columns):
    import pandas as pd
    df = pd.DataFrame(columns=columns)
    for f in in_files:
        print '^^^ ' + f
        d = pd.read_csv(f)
        df = df.merge(d, how='outer')
    df.to_csv(out_file, index=False)

stop_words_file = real_dir_path + '/stopwords.txt'
stopwords = {}.fromkeys([line.rstrip() for line in open(stop_words_file)]) #utf-8
stopwords_set = set(stopwords)

from pyltp import Segmentor, Postagger
segmentor = Segmentor()
segmentor.load('/root/git/ltp_data/cws.model')
#segmentor.load('/Users/a000/git/ltp_data/cws.model')
poser = Postagger()
poser.load('/root/git/ltp_data/pos.model')
#poser.load('/Users/a000/git/ltp_data/pos.model')

allow_pos_ltp = ('a', 'i', 'j', 'n', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'v', 'ws')
#使用哈工大pyltp分词, 过滤词性
def cut_pos_ltp(doc, filter_pos = True, allow_pos = allow_pos_ltp, remove_tags=True, return_str=True):
    s = ''.join(doc.split())  #去除空白符
    if remove_tags:
        soup = BeautifulSoup(s, 'lxml')
        s = soup.get_text().encode('utf-8')
    try:
        words = segmentor.segment(s)
    except:
        print type(s)
        print 's is {}'.format(s)
        print traceback.print_exc()

    words2 = []
    for w in words:
        if len(w.decode('utf-8')) > 1 and (w not in stopwords_set):
            words2.append(w)
    if not filter_pos:
        return ' '.join(words2)

    poses = poser.postag(words2)
    ss = []
    for i, pos in enumerate(poses):
        if pos in allow_pos:
            ss.append(words2[i])
    if return_str:
        return ' '.join(ss)
    else:
        return ss

#数据库查询从节点
#POSTGRE_HOST_QUERY = '120.27.162.201'
POSTGRE_HOST_QUERY = '10.47.54.32'
POSTGRE_DBNAME_QUERY = 'BDP'
#POSTGRES_QUERY = "postgresql://postgres:ly@postgres&2015@120.27.162.201:5432/BDP"
POSTGRES_QUERY = "postgresql://postgres:ly@postgres&2015@10.47.54.32:5432/BDP"
def get_postgredb_query():
    try:
        connection = psycopg2.connect(database=POSTGRE_DBNAME_QUERY, user=POSTGRE_USER, password=POSTGRE_PWD, host=POSTGRE_HOST_QUERY,)
        cursor = connection.cursor()
        return connection, cursor
    except:    #出现异常,再次连接
        try:
            time.sleep(2)
            connection = psycopg2.connect(database=POSTGRE_DBNAME_QUERY, user=POSTGRE_USER, password=POSTGRE_PWD, host=POSTGRE_HOST_QUERY,)
            cursor = connection.cursor()
            return connection, cursor
        except:
            traceback.print_exc()
            raise

channle_sql = "select ni.title, ni.content, ni.nid from info_news ni " \
              "inner join channellist_v2 c on ni.chid=c.id " \
              "inner join newslist_v2 nv on ni.nid=nv.nid " \
              "where c.cname=%s and nv.state=0 " \
              "order by ni.nid desc limit %s"
#########################  提供多进程版本  ####################################
#读取频道新闻
def coll_chnal(chname, num, to_csv=True, save_path=''):
    conn, cursor = get_postgredb_query()
    chnal = []
    nids = []
    docs = []
    cursor.execute(channle_sql, (chname, num))
    rows = cursor.fetchall()
    for row in rows:
        title = row[0]
        content_list = row[1]
        txt = ''
        for content in content_list:
            if 'txt' in content.keys():
                txt += content['txt'] + ' '   #unicode

        soup = BeautifulSoup(txt, 'lxml')
        txt = soup.get_text()
        total_txt = title + ' ' + txt.encode('utf-8')
        #去除三种特殊空格
        total_txt = total_txt.replace('\xe2\x80\x8b', '')
        total_txt = total_txt.replace('\xe2\x80\x8c', '')
        total_txt = total_txt.replace('\xe2\x80\x8d', '')

        chnal.append(chname)
        nids.append(row[2])
        docs.append(''.join(total_txt.split())) #split主要去除回车符\r, 否则pandas.read_csv出错
    if to_csv:
        data = {'chnl':chnal, 'nid':nids, 'doc':docs}
        df = pd.DataFrame(data, columns=('chnl', 'nid', 'doc'))
        df.to_csv(save_path, index=False)
    conn.close()


def coll_cut_chnal(chname, num, save_dir, cut_save_file):
    try:
        save_path = os.path.join(save_dir, chname+'_raw.csv')
        coll_chnal(chname, num, True, save_path)
        print '-------{} coll finish!'.format(chname)
        raw_df = pd.read_csv(save_path)
        docs_series = raw_df['doc']
        docs_series = docs_series.apply(cut_pos_ltp, (True, allow_pos_ltp, False))
        raw_df['doc'] = docs_series
        raw_df.to_csv(cut_save_file, index=False)
        print '**************{} cut finished! '.format(chname)
    except:
        traceback.print_exc()


def coll_cut_extract_multiprocess(chnl_num_dict,
                                  save_dir):
    from multiprocessing import Pool
    pool = Pool(30)
    chnl_cut_file = []
    for item in chnl_num_dict.items():
        chnl = item[0]
        num = item[1]
        cut_save_path = os.path.join(save_dir, chnl+'_cut.csv')
        chnl_cut_file.append(cut_save_path)
        pool.apply_async(coll_cut_chnal, args=(chnl, num, save_dir, cut_save_path))
    pool.close()
    pool.join()

    data_cut_path = os.path.join(save_dir, 'data_cut.csv')
    join_csv(chnl_cut_file, data_cut_path, columns=('chnl', 'nid', 'doc'))


if __name__ == "__main__":
    try:
        chnl_newsnum_dict = {'财经':100000, '互联网':100000, '健康':100000, '军事':100000,
                             '汽车':100000, '养生':100000, '影视':10000,
                             '游戏':100000, '育儿':100000, '体育':100000, '娱乐':100000}
        coll_cut_extract_multiprocess(chnl_newsnum_dict, './data/data_cut')
    except:
        traceback.print_exc()



