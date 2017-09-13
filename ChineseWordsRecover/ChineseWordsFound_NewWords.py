# -*- coding: utf-8 -*-
"""
andy@datassis.com
"""

import json
from collections import OrderedDict
import time
import collections
fdir_dic="C:\\Users\Administrator\Desktop\\data.json"
fdir_daily="C:\\Users\Administrator\Desktop\\data2.json"

def load(fdir):
    with open(fdir,'r',encoding='utf8') as json_file:
        data = json.load(json_file,object_pairs_hook=OrderedDict)
    return data

time1 = time.time()
old_dict = load(fdir_dic)    #有序字典
new_dict = load(fdir_daily)
time2 = time.time()
print('----------读取Json：%.2f秒--------'%(time2-time1))
dictwords = list(old_dict.keys())
new_dictwords = list(new_dict.keys())

newords = list(set(new_dictwords).difference(set(dictwords)))   #新词

# datassis_words_dic = list(set(new_dictwords).union(set(dictwords)))   #合并

dict_temp ={}
for i in range(len(newords)):
    dict_temp[newords[i]]=new_dict[newords[i]]
time3 = time.time()
print('----------捕捉新词：%.2f秒--------'%(time3-time2))
dictMerged=dict(old_dict, **dict_temp)  #字典合成
time4 = time.time()
print('----------字典合成：%.2f秒--------'%(time4-time3))
score_list =  sorted(dictMerged.items(), key=lambda item: item[1], reverse=True)
score_list_new =  sorted(dict_temp.items(), key=lambda item: item[1], reverse=True)
final_score = collections.OrderedDict()
newords_score = collections.OrderedDict()
for i in range(len(score_list)):
    final_score[score_list[i][0]]=score_list[i][1]
for i in range(len(score_list_new)):
    newords_score[score_list_new[i][0]]=score_list_new[i][1]
time5 = time.time()
print('----------按置信Value重排序：%.2f秒--------'%(time5-time4))

def store(data,fdir):
    with open(fdir, 'w',encoding='utf8') as json_file:
        json_file.write(json.dumps(data,ensure_ascii=False))

store(final_score,fdir='C:\\Users\Administrator\Desktop\words_dict.json')
store(newords_score,fdir='C:\\Users\Administrator\Desktop\\DailyWordsNew.json')
time6 = time.time()
print('----------本地化：%.2f秒--------'%(time6-time5))
















# def readWords(fdir):
#     f = codecs.open(fdir, 'r', encoding='utf8')
#     datassis_words = f.read()
#     f.close()
#     datassis_words = datassis_words.strip('[]')
#     datassis_words = re.findall(r'[^()]+', datassis_words)
#     datassis_words = [i for i in datassis_words if i != ', ']
#
#     return datassis_words
#
#
#
# datassis_dic = readWords(fdir_dic)
# datassis_daily = readWords(fdir_daily)
#
# words_dic =[]
# words_daily=[]
# for i in range(len(datassis_dic)):
#     words_dic.append(datassis_dic[i].split(',')[0].strip("'"))
# for i in range(len(datassis_daily)):
#     words_daily.append(datassis_daily[i].split(',')[0].strip("'"))
#
# newords = list(set(words_daily).difference(set(words_dic)))
# # datassis_words_dic = list(set(a).union(set(b)))
# print('------------新词有%s个-------------'%(len(newords)))
#
# write_to_txt("C:\\Users\Administrator\Desktop\\new_words.txt",str(newords))

