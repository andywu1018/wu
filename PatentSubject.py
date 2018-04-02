#*-utf8
#author:wudi@zjyiqiao.com

#项目说明：
#1：用户输入一段描述性文本，通过脚本解析语义，并在数据库中检索出用户“想要”的内容
#2：专利主题提取
import pandas as pd
import pymssql
import time
from multiprocessing import Process, Lock
from scipy import stats
from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from multiprocessing import Pool
from sklearn import preprocessing

# time1 = time.time()
# server = "172.16.5.45"
# user = "sa"
# password = "wlzx87811024"
# db = "IPOL"
#
# conn = pymssql.connect(server,user,password,db)
# cur = conn.cursor()
# # cur.execute('select top 1000 * from [dbo].[PATENT]')
# sqlcmd = 'select top 1000 * from [dbo].[PATENT]'
# #如果update/delete/insert记得要conn.commit()
# #否则数据库事务无法提交
# data = pd.read_sql(sqlcmd,conn)
# cur.close()
#
# NameOfPatent = data['MC']  #专利名称
# Summary = data['ZY']       #摘要
# Sovereignty = data['ZQX']  #主权项：需要保护的主要权利


import sys
import re
import jieba
jieba.add_word('靶材')
jieba.add_word('纯金属材料')
# jieba.load_userdict("newdict.txt")




def read_list(path):
    fp = open(path, 'r', encoding='utf8')
    lines = []
    for line in fp.readlines():
        line = line.strip()
        lines.append(line)
    fp.close()
    return lines
stopwords = read_list('D:\\2018年工作\PyProgram\stopwords.txt')
wordlist = read_list('D:\\2018年工作\PyProgram\patent.zh.simp.seg.txt')

def filewordProcess(content):
    wordlist = []
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'\n', ' ', content)
    content = re.sub(r'\t', ' ', content)
    re_chinese = re.compile(u'[\u4e00-\u9fa5]+')
    content = ''.join(re.findall(re_chinese, content)) + '\r\n'
    for seg in jieba.cut(content,cut_all=False):
        if seg not in stopwords:
            if seg != ' ':
                wordlist.append(seg)
    file_content = ' '.join(wordlist)
    return file_content,wordlist



def tfidf_mat(words_list,N_gram):  #N_gram :最大特征数
    freWord = CountVectorizer()
    transformer = TfidfTransformer(norm='l2',use_idf=True)  #欧几里德距离
    fre_matrix = freWord.fit_transform(words_list)
    tfidf = transformer.fit_transform(fre_matrix)

    feature_names = freWord.get_feature_names()  # 特征名
    freWordVector_df = pd.DataFrame(fre_matrix.toarray())  # 全词库 词频 向量矩阵
    tfidf_df = pd.DataFrame(tfidf.toarray())  # tfidf值矩阵
    #print(tfidf_df)
    # print freWordVector_df
    # tf-idf 筛选
    tfidf_sx_featuresindex = tfidf_df.sum(axis=0).sort_values(ascending=False)[:N_gram].index
    df_columns = pd.Series(feature_names)[tfidf_sx_featuresindex]
    freWord_tfsx_df = tfidf_df.ix[:, tfidf_sx_featuresindex] # tfidf法筛选后的idf矩阵

    #         print(df_columns[j], freWord_tfsx_df[i][j])
    return df_columns,freWord_tfsx_df  #column：词序列，fre：tfidf矩阵
    #print(freWord_tfsx_df)

#InfoTfidfList 是输入内容对应模型特征的词向量

#余弦
def cosVector(x,y):
    if (len(x)!=len(y)):
        print('error input,x and y is not in the same space')
        return
    result1=0.0
    result2=0.0
    result3=0.0
    for i in range(len(x)):
        result1+=x[i]*y[i]   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=y[i]**2
    return result1/((result2*result3)**0.5)#sum(Y*Y)
    #print(result1)
    #print(result2)
    #print(result3)
    # print("result is "+str(result1/((result2*result3)**0.5))) #结果显示
def MapSpearman(x):
    return stats.spearmanr(InfoTfidfList, x)
    # print(a)


# def MulThreading():
#     a=[]
#     for n in B:
#          a.append(tpool.apply(func=MapSpearman,args=(n,)))
#     print(max(a))
#     tpool.close()
#     tpool.join()

# TfidfArray = b.copy()\\
if __name__ == "__main__":
    SearchInfo = input("请描述检索内容：")
    INFO, INFO_LIST = filewordProcess(SearchInfo)
    from collections import Counter

    INFOBAG = Counter(INFO_LIST)
    wordlist_1 = wordlist[0:10000]
    N_gram = 100
    a, b = tfidf_mat(wordlist_1, N_gram)
    dict_new = {value: key for key, value in dict(a).items()}
    ret = [i for i in dict_new.keys() if i not in INFOBAG.keys()]
    ret = [dict_new[i] for i in ret]
    INFO_RET = dict(INFOBAG).fromkeys(ret, 0)
    intersection = [i for i in INFOBAG.keys() if i in dict_new.keys()]
    intersection = [dict_new[i] for i in intersection]
    INFO_INTERSEC = dict(INFOBAG).fromkeys(intersection, 1)
    Z = {**INFO_RET, **INFO_INTERSEC}
    InfoTfidfList = []
    for i in a.index.tolist():
        InfoTfidfList.append(Z[i])
    B = b.as_matrix()

    time1 = time.time()
    # tpool = Pool(8)
    result = list(map(MapSpearman,b.as_matrix()))
    # result = tpool.apply_async(MapSpearman,B)
    print(max(result))
    print(result.index(max(result)))
    # tpool.close()
    # tpool.join()
    # result.get()
    # MulThreading()
    time2 = time.time()
    print(time2 - time1)



