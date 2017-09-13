# -*- coding: utf-8 -*-
"""
andy@datassis.com
"""
#############--------------维护词典：衰退淘汰机制---------------################
import json
import time
import collections

def store(data,fdir):
    with open(fdir, 'w',encoding='utf8') as json_file:
        json_file.write(json.dumps(data,ensure_ascii=False))

def load(fdir):
    with open(fdir,'r',encoding='utf8') as json_file:
        data = json.load(json_file)
    return data

time1 =time.time()
print('--------------读取Json并重新赋值--------------')
fdir = "C:\\Users\Administrator\Desktop\\words_dict.json"
fdir_dailynew="C:\\Users\Administrator\Desktop\\DailyWordsNew.json"
fdir_daily = "C:\\Users\Administrator\Desktop\\sh_word.json"
fdir_out = "C:\\Users\Administrator\Desktop\\words_dict_days.json"
words_dict = load(fdir)
daily_dict = load(fdir_daily)
dailynew_dict = load(fdir_dailynew)
dictwords = list(words_dict.keys())
daily_dictwords = list(daily_dict.keys())
#逐日衰退机制：初始值：14
datassis_wordsdict_days = words_dict.fromkeys(words_dict.keys(),14)
dailywordsnew_days = dailynew_dict.fromkeys(dailynew_dict.keys(),13)
dailytotal_days = daily_dict.fromkeys(daily_dict.keys(),1)
time2 = time.time()
print('-------------------初始化完成:%.2f秒----------------'%(time2-time1))
#词频衰退机制：初始值：0
# datassis_wordsdict_freq = words_dict.fromkeys(words_dict.keys(),0)

#合并字典，相同key相加
def union_dict(*objs):
    _keys = set(sum([list(obj.keys()) for obj in objs], []))
    _total = {}
    for _key in _keys:
        _total[_key] = sum([obj.get(_key, 0) for obj in objs])
    return _total
print ('----------------进行衰退计算----------------')
miss_words = list(set(dictwords).difference(set(daily_dictwords)))
dict_temp ={}
for i in range(len(miss_words)):
    dict_temp[miss_words[i]]= -1

datassis_wordsdict_days = union_dict(datassis_wordsdict_days,dailywordsnew_days,dailytotal_days,dict_temp)
time3 =time.time()
print('----------计算完成:%.2f秒,进行本地化----------'%(time3-time2))
score_list =  sorted(datassis_wordsdict_days.items(), key=lambda item: item[1], reverse=True)
final_score = collections.OrderedDict()
for i in range(len(score_list)):
    final_score[score_list[i][0]]=score_list[i][1]


store(final_score,fdir_out)

print('====================任务完成====================')
