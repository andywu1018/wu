#*-utf8
#author:wudi@zjyiqiao.com


#encoding:utf-8
import gensim
import numpy as np
import jieba
from numba import vectorize
import re
from gensim import utils
from random import shuffle
import multiprocessing
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
# stop_text = open('stop_list.txt', 'r')
# stop_word = []
# for line in stop_text:
#     stop_word.append(line.strip())
# TaggededDocument = gensim.models.doc2vec.TaggedDocument
McCorpusFdir ="D:\\2018年工作\PyProgram\mc.zh.simp.seg_total.txt"
ZqxCorpusFdir = "D:\\2018年工作\PyProgram\zqx.zh.simp.seg_total.txt"
ZyCorpusFdir = "D:\\2018年工作\PyProgram\zy.zh.simp.seg_total.txt"

McModelFdir = "D:\\2018年工作\PyProgram\model_doc2vec_mc_300"
ZqxModelFdir = "D:\\2018年工作\PyProgram\model_doc2vec_zqx_300"
ZyModelFdir = "D:\\2018年工作\PyProgram\model_doc2vec_buildvocab_300"
#加载停用词
def read_list(path):
    fp = open(path, 'r', encoding='utf8')
    lines = []
    for line in fp.readlines():
        line = line.strip()
        lines.append(line)
    fp.close()
    return lines
stopwords = read_list('D:\\2018年工作\PyProgram\stopwords.txt')
#分词
def filewordProcess(content):
    wordlist = []
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'\n', ' ', content)
    content = re.sub(r'\t', ' ', content)
    re_chinese = re.compile(u'[\u4e00-\u9fa5]+')
    content = ''.join(re.findall(re_chinese, content))
    for seg in jieba.cut(content,cut_all=False):
        if seg not in stopwords:
            if seg != ' ':
                if len(seg)>=2:
                   wordlist.append(seg)
    # file_content = ' '.join(wordlist)
    return wordlist


import time
#加载模型
time2= time.time()
model_mc = Doc2Vec.load(McModelFdir)
model_zqx = Doc2Vec.load(ZqxModelFdir)
model_zy = Doc2Vec.load(ZyModelFdir)
time1 =time.time()
print(time1-time2)
#用户输入文本
text_test = input('请输入专利名称:')
#对文本分词
mc_raw = filewordProcess(text_test)
#用户输入文本
text_test = input('请输入专利摘要描述:')
#对文本分词
zy_raw = filewordProcess(text_test)
#用户输入文本
text_test = input('请输入专利权利描述:')
#对文本分词
zqx_raw = filewordProcess(text_test)

#特征合并与计算
def SearchProcess(topn):
    if mc_raw !='':
       inferred_vector_dm = model_mc.infer_vector(mc_raw)
       sims_mc = model_mc.docvecs.most_similar([inferred_vector_dm], topn=topn)
    else:
        sims_mc = []
    if zy_raw !='':
       inferred_vector_dm = model_zy.infer_vector(zy_raw)
       sims_zy = model_zy.docvecs.most_similar([inferred_vector_dm], topn=topn)
    else:
        sims_zy = []
    if zqx_raw !='':
       inferred_vector_dm = model_zqx.infer_vector(zqx_raw)
       sims_zqx = model_zqx.docvecs.most_similar([inferred_vector_dm], topn=topn)
    else:
        sims_zqx = []

    return sims_mc,sims_zy,sims_zqx

sims_mc ,sims_zy,sims_zqx = SearchProcess(topn=100)
print(sims_mc)
print(sims_zy)
print(sims_zqx)

from collections import Counter
dict_mc = {a[2:]:b/6 for a,b in sims_mc}
dict_zy = {a[2:]:b/3 for a,b in sims_zy}
dict_zqx = {a[2:]:b/2 for a,b in sims_zqx}

dict_patent = dict(Counter(dict_mc)+Counter(dict_zy)+Counter(dict_zqx))
dict_patent = sorted(dict_patent.items(),key = lambda x:x[1],reverse = True)

print(dict_patent)