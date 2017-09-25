import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')
import sys
import gensim
import jieba
import math
import pymysql
import pandas as  pd
import numpy as np

def bm25_score_prepare(data_frame):
    weight={}
    sum = 0
    for x in contents:
        sum+=len(x)
    avgdl = sum/N_doc
    dlavgdl=[]
    for x in contents:
        dlavgdl.append(len(x)/avgdl)
    for key in N_q.keys():
        weight_sorce = math.log((N_doc-N_q[key]+0.5)/(N_q[key]+0.5))
        weight[key] = weight_sorce



    return weight  ,dlavgdl

def bm25_score(data_frame):
    weight,dlavgdl = bm25_score_prepare(data_frame)
    score = np.zeros(N_doc)
    for i in range(N_doc):
        for key in N_q.keys():
            fi = contents[i].count(key)
            score[i] += weight[key]*((k1*fi)/(fi+k1*(1-b+b*dlavgdl[i])))
    data_frame['score'] += score

    return score,data_frame







if __name__ == '__main__':
    fdir = 'D:/搜索/'
    model = gensim.models.Word2Vec.load(fdir+'wiki.zh.text.model')
    askingwords = input('请输入搜索词：')
    getwords = list(jieba.cut(askingwords))
    # keywords = getwords.append(askingwords)
    conn = pymysql.connect(
        host='x',
        port=3306,
        user='x',
        passwd='x',
        db='most',
        charset='utf8')
    features = ['ID_relative','title','content','topic']
    frame = []
    N_q = {}
    for word in getwords:
        try:
            words = model.wv.most_similar(word,topn=3)
            for t in words:
                print(t[0], t[1])
                sqlcmd = "select * from %s where content like '%%%%%s%%%%'" % ('similar_test', t[0])
                data = pd.read_sql(sqlcmd,conn)[features]
                N_q[str(t[0])]=len(data)
                data['score'] = 0
                frame.append(data)

        except:
            print('%s Not exist in the dict'%(word))
    for askingword in getwords:
        try:
            sqlcmd_askingword = "select * from %s where content like '%%%%%s%%%%'" % ('similar_test', askingword)
            data = pd.read_sql(sqlcmd_askingword, conn)[features]
            N_q[str(askingword)] = len(data)
            data['score'] = 5
            frame.append(data)
        except:
            print('%s Not exist in the dict' % (askingword))

    try:
        sqlcmd_askingwords = "select * from %s where content like '%%%%%s%%%%'" % ('similar_test', askingwords)
        data = pd.read_sql(sqlcmd_askingwords, conn)[features]
        N_q[str(askingwords)] = len(data)
        data['score'] = 10
        frame.append(data)
        data_frame = pd.concat(frame,ignore_index=True)
        data_frame = data_frame.drop_duplicates()
        data_frame = data_frame.reset_index(drop=True)

        # print(data_frame)


        # opencc - i wiki.zh.txt - o wiki.zh.simp.txt - c t2s.json
    except:
        print('%s Not exist in the dict' % (askingwords))

    N_doc = len(data_frame)
    k1=2
    b=0.75
    contents = data_frame['content']

    score,data_frame=bm25_score(data_frame)
    data_frame= data_frame.sort_values(by='score',ascending=False)
    print(data_frame)
