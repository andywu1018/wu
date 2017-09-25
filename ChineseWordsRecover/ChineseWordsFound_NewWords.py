# -*- coding: utf-8 -*-
"""
andy@datassis.com
"""

import json
from collections import OrderedDict
import time
import collections
import pymysql



content_list = {'bj': 0, 'ah': 0,'cq': 0,'fj': 0,'gd': 0,'gs': 0,'gx': 0,'gz': 0,'ha': 0,'hb': 0,'he': 0,'hi': 0,'hl': 0,'hn': 0,
                'jl': 0,'js': 0,'jx': 0,'ln': 0,'nm': 0,'nx': 0,'qh': 0,'sc': 0,'sd': 0,'sh': 0,'sn': 0,'state': 0,
                    'sx': 0,'tj': 0,'xj': 0,'xz': 0,'yn': 0,'zj': 0}
def load(fdir):
    with open(fdir,'r',encoding='utf8') as json_file:
        data = json.load(json_file,object_pairs_hook=OrderedDict)
    return data

def store(data, fdir):
    with open(fdir, 'w', encoding='utf8') as json_file:
        json_file.write(json.dumps(data, ensure_ascii=False, indent=4))

def insertSql(name,prob,table_name,spell):
    try:
        cur = conn.cursor()
        cur.execute(
            "insert into zz_dailywordsnew (char_name,char_prob,province,spell) values('%s','%s','%s','%s');" % (name,prob,table_name,spell
              ))
        print('insert yet')
        cur.close()
    except:
        print('insert error')

def NewWord(fdir_dic,fdir_daily,fdir_out):
    time2 = time.time()
    new_dict = load(fdir_daily)
    dictwords = list(old_dict.keys())
    new_dictwords = list(new_dict.keys())

    newords = list(set(new_dictwords).difference(set(dictwords)))  # 新词

    # datassis_words_dic = list(set(new_dictwords).union(set(dictwords)))   #合并

    dict_temp = {}
    for i in range(len(newords)):
        dict_temp[newords[i]] = new_dict[newords[i]]
    time3 = time.time()
    print('----------捕捉新词：%.2f秒--------' % (time3 - time2))
    # dictMerged=dict(old_dict, **dict_temp)  #字典合成
    # time4 = time.time()
    # print('----------字典合成：%.2f秒--------'%(time4-time3))
    # score_list =  sorted(dictMerged.items(), key=lambda item: item[1], reverse=True)
    score_list_new = sorted(dict_temp.items(), key=lambda item: item[1], reverse=True)
    # final_score = collections.OrderedDict()
    newords_score = collections.OrderedDict()
    # for i in range(len(score_list)):
    #     final_score[score_list[i][0]]=score_list[i][1]
    for i in range(len(score_list_new)):
        newords_score[score_list_new[i][0]] = score_list_new[i][1]
    time5 = time.time()
    print('----------按置信Value重排序：%.2f秒--------' % (time5 - time3))

    # store(final_score,fdir='C:\\Users\Administrator\Desktop\words_dict.json')
    store(newords_score,fdir_out)
    time6 = time.time()
    print('----------本地化：%.2f秒--------' % (time6 - time5))


import re
def convertCN2Spell(ch, max = 7):
    """
        该函数通过输入汉字返回其拼音，数字和字母会被跳过
    """
    length = len('迪') #测试汉字占用字节数，utf-8，汉字占用3字节.bg2312，汉字占用2字节
    # intord = ord(ch[0:1])
    ret = ""
    regularCHN = re.compile(r"[\u4e00-\u9fff]{1}")
    allCHNList = regularCHN.findall(ch)

    allCHNList = allCHNList[0:max]

    # if (intord >= 48 and intord <= 57):
    # 	ret = ret + ch[0:1]
    # if (intord >= 65 and intord <=90 ) or (intord >= 97 and intord <=122):
    # 	ret = ret + ch[0:1].lower()
    # ch = ch[0:length] #多个汉字只获取第一个

    tempDist = {}
    try:
        with open('C:\\Users\Administrator\Desktop\convertCN2Spell\convert-utf-8.txt', 'r', encoding='utf8') as f:
            for line in f:
                for chStr in allCHNList:
                    if chStr in line:
                        if line.find(","):
                            # print(line.find(','))
                            line = line[0:len(line) - 1]  # -2是txt编码问题
                            # print(line)
                        s = line.split(",")[0]  # 处理多音字情况
                        # print s, len(s)
                        tempDist[chStr] = s[length:-1]  # -1是去掉音调
                        # print(tempDist)
                        # print tempDist[ch]

        for origin in allCHNList:
            ret = ret + tempDist[origin]
    except:
        ret = ''

    return ret

fdir_dic = "/andy/ChineseWordSegment@datassis.com/words_dict.json"
old_dict = load(fdir_dic)  # 有序字典
for i in content_list:
    table_name = str(i)
    fdir_daily = "/andy/ChineseWordSegment@datassis.com/words/words_daily_%s.json"%(table_name)
    fdir_out  ='/andy/ChineseWordSegment@datassis.com/words/DailyWordsNew_%s.json'%(table_name)
    NewWord(fdir_dic,fdir_daily,fdir_out)
    conn = pymysql.connect(host='10.2.2.250',
        port=3306,
        user='wd',
        passwd='123456',
        db='policies',
        charset='utf8')
    dailynew = load(fdir_out)
    for j  in dailynew:
        prob = dailynew[j]
        try:
           spell = convertCN2Spell(str(j))
           insertSql(j,prob,table_name,spell)
        except:
            print('none')

    conn.close()

print("done!")


