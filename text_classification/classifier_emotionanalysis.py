# -*- coding: utf-8 -*-
# author@datassis.com
from ttexr import tfidf_mat
from ttexr import filewordProcess
import json
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import pandas as pd
from math import exp
import jieba
jieba.add_word('双创')
jieba.add_word('创客')
import jieba.posseg
import pickle
import pickle
import pymysql
import re
import dbm
from collections import Counter
import ttexr
import codecs
import time
from sklearn.externals import joblib
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


def read_list(path):
    fp = open(path, 'r', encoding='utf8')
    lines = []
    for line in fp.readlines():
        line = line.strip()
        lines.append(line)
    fp.close()
    return lines
farming = read_list('D:\文本分类\标题分类识别字符\牧业.txt')
manufacture = read_list('D:\文本分类\标题分类识别字符\制造业.txt')
architecture = read_list('D:\文本分类\标题分类识别字符\建筑业.txt')
poor = read_list('D:\文本分类\标题分类识别字符\扶贫.txt')
finance = read_list('D:\文本分类\标题分类识别字符\经济金融.txt')
bigdata = read_list('D:\文本分类\标题分类识别字符\大数据.txt')
science_sent = read_list('D:\文本分类\标题分类识别字符\科技特派员.txt')
science_reward = read_list('D:\文本分类\标题分类识别字符\科技成果.txt')
environment = read_list('D:\文本分类\标题分类识别字符\环保.txt')
agriculture = read_list('D:\文本分类\标题分类识别字符\农业.txt')
foresty = read_list('D:\文本分类\标题分类识别字符\林业.txt')
fishing = read_list('D:\文本分类\标题分类识别字符\渔业.txt')
retail = read_list('D:\文本分类\标题分类识别字符\批发零售.txt')
transport = read_list('D:\文本分类\标题分类识别字符\交通运输.txt')
food_safety = read_list('D:\文本分类\标题分类识别字符\食品安全.txt')
edu = read_list('D:\文本分类\标题分类识别字符\教育.txt')
meeting = read_list('D:\文本分类\标题分类识别字符\会议活动.txt')
medic = read_list('D:\文本分类\标题分类识别字符\医疗.txt')
statute = read_list('D:\文本分类\标题分类识别字符\法治法规.txt')
livelihood = read_list('D:\文本分类\标题分类识别字符\民生活动.txt')
xingchuang = read_list('D:\文本分类\标题分类识别字符\星创天地.txt')
gov = read_list('D:\文本分类\标题分类识别字符\党政工作.txt')
nkc1 = read_list('D:\文本分类\训练数据3\农科创.txt')

def getSource_split(forum): #分割板块字符串，有两种分隔符
    split_list = []
    if '_' in forum:
        split_list = forum.split('_')
    elif '-' in forum:
        split_list = forum.split('-')
    return split_list

def get_keys(d, value):
    return [k for k,v in d.items() if v == value]

def spilit_heading(heading,forum):
    if heading !='':
        theme1 = ','.join(np.unique(re.findall('“(.*?)”',heading)))
        theme2 = ','.join(np.unique(re.findall('调研(.*?)情况',heading)))
        theme3 = ','.join(np.unique(re.findall('调研(.*?)工作', heading)))
        theme4 = ','.join(np.unique(re.findall('调研(.*?)项目', heading)))
        theme5 = ','.join(np.unique(re.findall('指导(.*?)建设', heading)))
        theme6 = ','.join(np.unique(re.findall('开展(.*?)调研', heading)))
        theme7 = ','.join(np.unique(re.findall('指导(.*?)工作', heading)))
        theme8 = ','.join(np.unique(re.findall('视察(.*?)工作', heading)))
        # theme10 = ','.join(np.unique(re.findall('(.*?)建设', heading)))
        theme11 = ','.join(np.unique(re.findall('针对(.*?)开展', heading)))
        theme12 = ','.join(np.unique(re.findall('调研(.*?)现状', heading)))
        theme13 = ','.join(np.unique(re.findall('深入(.*?)走访', heading)))
        listtheme = [theme1,theme2,theme3,theme4,theme5,theme6,theme7,theme8,theme11,theme12,theme13]
        while '' in listtheme:
            listtheme.remove('')
        # if listtheme ==[]:
        #     listtheme = [heading[heading.rfind('强调')+2:]]
        # if listtheme ==[]:
        #     listtheme = [heading[heading.rfind('调研')+2:]]
        length = 100
        min = ''
        for x in listtheme:
                if len(x)<length:
                    length = len(x)
                    min = x
        theme = min
        sheng_forum = ''.join(list((set(getSource_split(forum)).union(set(province_ssx))) ^ (set(getSource_split(forum)) ^ set(province_ssx))))
        shi_forum = ''.join(list((set(getSource_split(forum)).union(set(city_ssx))) ^ (set(getSource_split(forum)) ^ set(city_ssx))))
        qu_forum = ''.join(list((set(getSource_split(forum)).union(set(district_ssx))) ^ (set(getSource_split(forum)) ^ set(district_ssx))))
        spilit_dict = {}
        word_list = []
        for word in jieba.cut(heading,cut_all=False):
            word_list.append(word)
        for seg in jieba.posseg.cut(heading):
            # spilit_dict = {}
            spilit_dict[seg.word] = seg.flag
        # spilit_list=','.join(spilit_list)
        # address = get_keys(spilit_dict,'ns')
        name = get_keys(spilit_dict, 'nr')
        sheng = list((set(word_list).union(set(province_ssx))) ^ (set(word_list) ^ set(province_ssx)))
        if list((set(sheng).union(set(name))) ^ (set(sheng) ^ set(name))) != []:
            name.remove(sheng)
        sheng = ','.join(sheng)
        if sheng == '':
            sheng = sheng_forum
        shi = list((set(word_list).union(set(city_ssx))) ^ (set(word_list) ^ set(city_ssx)))
        if list((set(shi).union(set(name))) ^ (set(shi) ^ set(name))) != []:
            name.remove(shi)
        shi = ','.join(shi)
        if shi == '':
            shi = shi_forum
        qu = list((set(word_list).union(set(district_ssx))) ^ (set(word_list) ^ set(district_ssx)))
        if list((set(qu).union(set(name))) ^ (set(qu) ^ set(name))) != []:
            name.remove(qu)
        name = ','.join(name)
        qu = ','.join(qu)
        if qu == '':
            qu = qu_forum
        act = ','.join(get_keys(spilit_dict,'vn'))
        # print(heading)
        # print(spilit_dict)
        print("地点：省：\033[1;34m %s \033[0m市：\033[1;36m %s \033[0m区：\033[1;31m %s \033[0m人物:\033[1;35m %s \033[0m行为:\033[1;32m %s \033[0m主题:\033[1;36m %s \033[0m"%(sheng,shi,qu,name,act,theme))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    return theme,sheng,shi,qu,name,act

def tfidf_mat_pre(pre_text):
    words_list_pre = words_list
    filename_list_pre = filename_list
    category_list_pre = category_list
    pre_text = filewordProcess(pre_text)
    words_list_pre.append(pre_text)
    pre_file_name = '预测样本.txt'
    filename_list_pre.append(pre_file_name)
    pre_category_name = 'Unknown'
    category_list_pre.append(pre_category_name)
    tfidf_df_1_pre, df_columns_pre = tfidf_mat(words_list_pre, filename_list_pre, category_list_pre)
    tfidf_df_v = tfidf_df_1_pre.iloc[-1:]
    # print(tfidf_df_v, df_columns_pre)
    ch2 = SelectKBest(chi2, k=5000)
    nolabel_feature_pre = [x for x in tfidf_df_v.columns if x not in ['label']]
    miss_feature = [l for l in nolabel_feature if l not in nolabel_feature_pre]
    #print('丢失特征：%s >>>>>丢失个数:%s' % (miss_feature, len(miss_feature)))
    aa = {'预测样本.txt': np.zeros(len(miss_feature))}
    tfidf_df_v = tfidf_df_v.join(pd.DataFrame(aa.values(), columns=miss_feature, index=aa.keys()))
    # temp = tfidf_df_v.copy()
    # for i in range(len(temp.columns)):
    #     if temp.columns[i] not in nolabel_feature:
    #         tfidf_df_v = tfidf_df_v.drop(temp.columns[i], axis=1)
    ch2_sx_np = tfidf_df_v[nolabel_feature_pre].values
    return ch2_sx_np

def filewordprocess(content):
    wordlist = []
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(r'\d', ' ', content)
    content = re.sub(r'\n', ' ', content)
    content = re.sub(r'\t', ' ', content)
    content = re.sub(r'[a-zA-Z]', ' ', content)
    for seg in jieba.cut(content,cut_all=True):
        if seg not in stopwords:
            if seg != ' ':
                wordlist.append(seg)
    return wordlist

def getSource(forum,source):
    flag = 0
    var_level = ''
    var_source = source
    if len(getSource_split(forum)) > 1:
        split_source = getSource_split(forum)[-2]# 倒数第二个字段是来源，找规律，你懂得
        # print(table, data.index[i], split_source)
        if source == '国务院' or split_source == '国务院':  # 如果来源是国务院，板块倒数第二个字段也是国务院，则级别是国
            var_source = '国务院'
            var_level = '国务院'
            flag = 1
        elif getSource_split(forum)[0] == '国家' and split_source != '国务院':  # 板块第一个字段是国家，倒数第二个字段不是国务院，则级别是部委
            var_level = '部委'
            if source == '' or regex.search(source):
                var_source = split_source
            flag = 1
        elif source == '' or regex.search(source):
            var_source = split_source
    if flag == 0:  # 如果flag为1，则不再判断省市区
        # 判别省、市、区
        if province != '':
            if city != '':
                if qu != '':
                    var_level = '县级'
                else:
                    var_level = '市级'
            else:
                var_level = '省级'
        else:  # 省字段为空，则从板块中判断
            if len(getSource_split(forum)) == 2 or len(getSource_split(forum)) == 3:  # 板块中有两个字段或三个字段，就是省级别
                var_level = '省级'
            elif len(getSource_split(forum)) == 4:  # 板块中有4个字段，市级别
                var_level = '市级'
            elif len(getSource_split(forum)) == 5:  # 板块中有5个字段，区级别
                var_level = '县级'


    return  var_source,var_level

def classifier_pre(heading, content, source):
    # count=0
    heading_list = []
    clf = joblib.load("D:\文本分类\\train_model.m")
    for seg in jieba.cut(heading, cut_all=True):
        if seg != ' ':
            heading_list.append(seg)
    heading = ' '.join(heading_list)
    heading = heading.split(sep=' ')
    classifier = []
    details = '基于标题分类'
    if True in  list(map(lambda x:x in farming,heading)):
        classifier.append('牧业')
    if True in list(map(lambda x:x in manufacture,heading)):
        classifier.append('制造业')
    if True in list(map(lambda x:x in architecture,heading)):
        classifier.append('建筑业')
    if True in list(map(lambda x:x in poor,heading)):
        classifier.append('扶贫')
    if True in list(map(lambda x:x in finance,heading)):
        classifier.append('经济金融')
    if True in list(map(lambda x:x in bigdata,heading)):
        classifier.append('大数据')
    if True in list(map(lambda x:x in science_sent,heading)):
        classifier.append('科技特派员')
    if True in list(map(lambda x:x in science_reward,heading)):
        classifier.append('科技成果')
    if True in list(map(lambda x:x in environment,heading)):
        classifier.append('环保')
    if True in list(map(lambda x:x in agriculture,heading)):
        classifier.append('农业')
    if True in list(map(lambda x:x in foresty,heading)):
        classifier.append('林业')
    if True in list(map(lambda x:x in fishing,heading)):
        classifier.append('渔业')
    if True in list(map(lambda x:x in retail,heading)):
        classifier.append('批发零售')
    if True in list(map(lambda x:x in transport,heading)):
        classifier.append('交通运输')
    if True in list(map(lambda x:x in food_safety,heading)):
        classifier.append('食品安全')
    if True in list(map(lambda x:x in edu,heading)):
        classifier.append('教育')
    if True in list(map(lambda x:x in meeting,heading)):
        classifier.append('会议活动')
    if True in list(map(lambda x:x in medic,heading)):
        classifier.append('医疗')
    if True in list(map(lambda x:x in statute,heading)):
        classifier.append('法治法规')
    if True in list(map(lambda x:x in livelihood,heading)):
        classifier.append('民生活动')
    if True in list(map(lambda x:x in xingchuang,heading)):
        classifier.append('星创天地')
    if True in list(map(lambda x:x in gov,heading)):
        classifier.append('党政工作')
    if len(classifier)>=3 and '党政工作' in classifier:
        classifier.remove('党政工作')
    classifier = ','.join(np.unique(classifier))
    while (classifier == ''):
        details = '基于关键字分类'
        print('标题分类：未知>>>>>进行关键字分类>>>>>>')
        classifier = []
        words_list_topic = []
        words_list_topic.append(filewordProcess(content))
        topic_words = get_top_words(3, words_list_topic)
        topic_words = topic_words.split(" ")
        if True in list(map(lambda x: x in farming, topic_words)):
            classifier.append('牧业')
        if True in list(map(lambda x: x in manufacture, topic_words)):
            classifier.append('制造业')
        if True in list(map(lambda x: x in architecture, topic_words)):
            classifier.append('建筑业')
        if True in list(map(lambda x: x in poor, topic_words)):
            classifier.append('扶贫')
        if True in list(map(lambda x: x in finance, topic_words)):
            classifier.append('经济金融')
        if True in list(map(lambda x: x in bigdata, topic_words)):
            classifier.append('大数据')
        if True in list(map(lambda x: x in science_sent, topic_words)):
            classifier.append('科技特派员')
        if True in list(map(lambda x: x in science_reward, topic_words)):
            classifier.append('科技成果')
        if True in list(map(lambda x: x in environment, topic_words)):
            classifier.append('环保')
        if True in list(map(lambda x: x in agriculture, topic_words)):
            classifier.append('农业')
        if True in list(map(lambda x: x in foresty, topic_words)):
            classifier.append('林业')
        if True in list(map(lambda x: x in fishing, topic_words)):
            classifier.append('渔业')
        if True in list(map(lambda x: x in retail, topic_words)):
            classifier.append('批发零售')
        if True in list(map(lambda x: x in transport, topic_words)):
            classifier.append('交通运输')
        if True in list(map(lambda x: x in food_safety, topic_words)):
            classifier.append('食品安全')
        if True in list(map(lambda x: x in edu, topic_words)):
            classifier.append('教育')
        if True in list(map(lambda x: x in meeting, topic_words)):
            classifier.append('会议活动')
        if True in list(map(lambda x: x in medic, topic_words)):
            classifier.append('医疗')
        if True in list(map(lambda x: x in statute, topic_words)):
            classifier.append('法治法规')
        if True in list(map(lambda x: x in livelihood, topic_words)):
            classifier.append('民生活动')
        if True in list(map(lambda x: x in xingchuang, topic_words)):
            classifier.append('星创天地')
        if True in list(map(lambda x: x in gov, topic_words)):
            classifier.append('党政工作')
        classifier = ','.join(np.unique(classifier))
        while (classifier == ''):
            print('关键字分类：未知>>>>>进行来源分类>>>>>>')
            classifier = []
            details = '基于来源分类'
            for j in range(len(source)):
                if source[j] in ['农', '粮']:
                    classifier.append('农业')
                elif source[j] in ['林']:
                    classifier.append('林业')
                elif source[j] in ['牧', '畜', '养', '殖']:
                    classifier.append('牧业')
                elif source[j] in ['药', '医']:
                    classifier.append('医疗')
                elif source[j] in ['食']:
                    classifier.append('食品安全')
                elif source[j] in ['财', '经', '银', '行']:
                    classifier.append('经济金融')
                elif source[j] in ['环', '保']:
                    classifier.append('环保')
                elif source[j] in ['交', '车']:
                    classifier.append('交通运输')
            classifier = ','.join(np.unique(classifier))
            while (classifier == ''):
                print('来源分类：未知>>>>>进行内容分类>>>>>>')
                ch2_sx_np = tfidf_mat_pre(content)
                p = clf.predict_proba(ch2_sx_np).flatten()
                # print(p)
                top_index, top_prob = top3(p)
                # print(top_index,top_prob)
                first = top_prob[0]
                second = top_prob[1]
                third = top_prob[2]
                if first <= 0.4:
                    classifier = '其他'
                    details = '无法判别'
                else:
                    details = '基于内容分类'
                    if first / second >= 2:
                        classifier = '%s' % (category_list_v[top_index[0]])
                    else:
                        if first / third >= 2:
                            classifier = '%s,%s' % (category_list_v[top_index[0]], category_list_v[top_index[1]])
                        else:
                            classifier = '%s,%s,%s' % (category_list_v[top_index[0]], category_list_v[top_index[1]],
                                                       category_list_v[top_index[2]])
    print(classifier, details)
    return classifier, details


def encode_data(data):
    try:
        result = data.encode('utf-8')
    except:
        result = str(data)
    return result


# ===============================去停用词===================================
def del_stopwords(seg_sent):
    stopwords_ = codecs.open('D:\文本分类\文本情感分析\\news_analysis\stop_words.txt', 'r', encoding='utf8')
    stopwords = stopwords_.read()
    stopwords_.close()
    new_sent = []
    for word in seg_sent:
        if word in stopwords:
            continue
        else:
            new_sent.append(word)
    return new_sent


def segmentation(sentence):
    seg_list = jieba.cut(sentence)
    seg_result = []
    for w in seg_list:
        seg_result.append(w)
    return seg_result


# ==============================分句===================================
def cut_sentence(words):
    #	words = words.decode('utf8')
    start = 0
    i = 0
    token = 'meaningless'
    sents = []
    punt_list = (',.!?;~，。！？；～… ')
    for word in words:
        if word not in punt_list:  # 如果不是标点符号
            i += 1
            token = list(words[start:i + 2]).pop()  # 返回列表的最后一项
        elif word in punt_list and token in punt_list:  # 处理省略号（普通标点是1个，而省略号有一串）
            i += 1
            token = list(words[start:i + 2]).pop()
        else:
            sents.append(words[start:i + 1])  # 断句
            start = i + 1
            i += 1
    if start < len(words):  # 处理最后的部分
        sents.append(words[start:])
    return sents


# ==============================程度副词匹配=====================
def match(word, sentiment_value):
    if word in mostdict:
        sentiment_value *= 2.0
    elif word in verydict:
        sentiment_value *= 1.75
    elif word in moredict:
        sentiment_value *= 1.5
    elif word in ishdict:
        sentiment_value *= 1.2
    elif word in insufficientdict:
        sentiment_value *= 0.5
    elif word in inversedict:
        sentiment_value *= -1
    return sentiment_value


# =================情感得分的最后处理，防止出现负数===================
def transform_to_positive_num(poscount, negcount):
    pos_count = 0
    neg_count = 0
    if poscount < 0 and negcount >= 0:
        neg_count += negcount - poscount
        pos_count = 0
    elif negcount < 0 and poscount >= 0:
        pos_count = poscount - negcount
        neg_count = 0
    elif poscount < 0 and negcount < 0:
        neg_count = -poscount
        pos_count = -negcount
    else:
        pos_count = poscount
        neg_count = negcount
    return (pos_count, neg_count)


# =====================求单条新闻的情感倾向总得分===================
def single_review_sentiment_score(content):
    single_review_senti_score = []
    possen = []
    negsen = []
    cuted_review = cut_sentence(content)
    for sent in cuted_review:
        seg_sent = segmentation(sent)  # 分词
        seg_sent = del_stopwords(seg_sent)[:]
        i = 0  # 记录扫描到的词的位置
        s = 0  # 记录情感词的位置
        poscount = 0  # 记录该分句中的积极情感得分
        negcount = 0  # 记录该分句中的消极情感得分
        for word in seg_sent:  # 逐词分析
            if word in posdict:  # 如果是积极情感词
                poscount += posemotion[posdict.index(word)]  # 积极得分+1
                possen.append(possentiments[posdict.index(word)])
                for w in seg_sent[s:i]:
                    poscount = match(w, poscount)
                s = i + 1  # 记录情感词的位置变化

            elif word in negdict:  # 如果是消极情感词
                negcount += negemotion[negdict.index(word)]
                negsen.append(negsentiments[negdict.index(word)])
                for w in seg_sent[s:i]:
                    negcount = match(w, negcount)
                s = i + 1
            elif word == "！":  # or word == "!".decode('utf-8') //不知道这样写干嘛用
                for w2 in seg_sent[::-1]:  # 倒序扫描感叹号前的情感词，发现后权值+2，然后退出循环
                    if w2 in posdict:  # ！前的情感词双倍效果
                        poscount += 2 * posemotion[posdict.index(w2)]
                        break
                    elif w2 in negdict:
                        negcount += 2 * negemotion[negdict.index(w2)]
                        break
            i += 1
        single_review_senti_score.append(transform_to_positive_num(poscount, negcount))  # 对得分做最后处理
    pos_result, neg_result = 0, 0  # 分别记录积极情感总得分和消极情感总得分
    for res1, res2 in single_review_senti_score:  # 每个分句循环累加
        pos_result += res1
        neg_result += res2
    posC = Counter(possen)
    negC = Counter(negsen)
    happy = posC['PA'] + posC['PE']
    fine = posC['PD'] + posC['PH'] + posC['PG'] + posC['PB'] + posC['PH']
    fury = negC['NA']
    wane = negC['NB'] + negC['NJ'] + negC['NH'] + negC['PF']
    fear = negC['NI'] + negC['NC'] + negC['NG']
    vice = negC['NE'] + negC['ND'] + negC['NN'] + negC['NK'] + negC['NL']
    shock = negC['PC']
    if pos_result+neg_result>0:
        result = pos_result / (pos_result + neg_result)
        result = 1.12 - exp(-1.6 * result)  # 该条新闻情感的最终得分
        result = round(result, 3)
        if result <= 0.4:
            emotion = '较大利空'
        if result > 0.4 and result <= 0.5:
            emotion = '较小利空'
        if result > 0.5 and result <= 0.6:
            emotion = '中性'
        if result > 0.6 and result <= 0.9:
            emotion = '较小利好'
        if result > 0.9:
            emotion = '较大利好'
        emotion_num = result
        emotion_result = emotion
    else:
        emotion_num = 0.12
        emotion_result = '中性'
    return emotion_result,emotion_num  # ,pos_result,neg_result,happy,fine,fury,wane,fear,vice,shock


#============================构建主题词=================================
def tfidf_mat_topic(words_list):
    freWord = CountVectorizer(stop_words='english', decode_error='ignore', )
    transformer = TfidfTransformer()
    fre_matrix = freWord.fit_transform(words_list)
    tfidf = transformer.fit_transform(fre_matrix)
    feature_names = freWord.get_feature_names()  # 特征名
    freWordVector_df = pd.DataFrame(fre_matrix.toarray())  # 全词库 词频 向量矩阵
    tfidf_df = pd.DataFrame(tfidf.toarray())  # tfidf值矩阵
    # tfidf_sx_featuresindex = tfidf_df.sum(axis=0).sort_values(ascending=False)[:30000].index
    # freWord_tfsx_df = freWordVector_df.ix[:,tfidf_sx_featuresindex] # tfidf法筛选后的词向量矩阵
    df_columns = pd.Series(feature_names)
    tfidf_df_2 = tfidf_df  # .apply(normalization)            #训练过程采用多项式模型精度更高
    tfidf_df_2.columns = df_columns
    # tfidf_df_1.index = filename_list
    return tfidf_df_2


def get_top_words(n_top_words, words_list):
    a = []
    tfidf_df_3 = tfidf_mat_topic(words_list)
    lda = LatentDirichletAllocation(learning_method='online', random_state=0, max_iter=50, learning_offset=10,
                                    n_topics=1)
    model = lda.fit(tfidf_df_3)
    for topic_idx, topic in enumerate(model.components_):

        a.append(" ".join([tfidf_df_3.columns[i]
                           for i in topic.argsort()[:-n_top_words - 1:-1]]))
    topic_words = ''.join(a)
    return topic_words

def geography_lacation(heading,content):
    heading_seg = filewordprocess(heading)
    content_seg = filewordprocess(content)
    nkc = []
    if True in list(map(lambda x: x in nkc1, content_seg)) or True in list(map(lambda x: x in nkc1, heading_seg)):
        nkc.append('是')
    else:
        nkc.append('否')
    nkc = ''.join(np.unique(nkc))
    print(list((set(content_seg).union(set(nkc1))) ^ (set(content_seg) ^ set(nkc1))),list((set(heading_seg).union(set(nkc1))) ^ (set(heading_seg) ^ set(nkc1))))

    qu_intersection_list = []
    city_intersection_list = []
    province_intersection_list = []
    for i in range(len(qu_list)):
        qu_intersection_heading = list((set(heading_seg).union(set(qu_list[i]))) ^ (set(heading_seg) ^ set(qu_list[i])))
        qu1 = ''.join(qu_intersection_heading)
        qu_intersection_list.append(qu1)
    District = ''.join(qu_intersection_list)
    if (District ==''):
        for i in range(len(qu_list)):
            qu_intersection_content = list(set(content_seg).intersection(set(qu_list[i])))
            qu1 = ''.join(qu_intersection_content)
            qu_intersection_list.append(qu1)
    District = ''.join(np.unique(qu_intersection_list))

    city_intersection_heading = list((set(heading_seg).union(set(city_list))) ^ (set(heading_seg) ^ set(city_list)))
    city1 = ''.join(city_intersection_heading)
    city_intersection_list.append(city1)
    City= ''.join(city_intersection_list)
    if (City ==''):
        city_intersection_content = list((set(content_seg).union(set(city_list))) ^ (set(content_seg) ^ set(city_list)))
        city1 = ''.join(city_intersection_content)
        city_intersection_list.append(city1)
    City = ''.join(np.unique(city_intersection_list))

    province_intersection_heading = list((set(heading_seg).union(set(province_list))) ^ (set(heading_seg) ^ set(province_list)))
    province1 = ''.join(province_intersection_heading)
    province_intersection_list.append(province1)
    Province = ''.join(province_intersection_list)
    if (Province == ''):
        province_intersection_content = list((set(content_seg).union(set(province_list))) ^ (set(content_seg) ^ set(province_list)))
        province1 = ''.join(province_intersection_content)
        province_intersection_list.append(province1)
    Province= ''.join(np.unique(province_intersection_list))
    # print(Province,City,District)

    return District,City,Province,nkc

def classifier_insertSql(content, heading, source,forum, j):
    words_list_topic = []
    words_list_topic.append(filewordProcess(content))
    classifier, details = classifier_pre(heading, content, source)
    District , City , Province,nkc = geography_lacation(heading,content)
    emotion_result,emotion_num = single_review_sentiment_score(content)
    topic_words = get_top_words(3, words_list_topic)
    var_source , var_level = getSource(forum,source)
    theme,sheng,shi,district,name,act = spilit_heading(heading,forum)
    try:
        # j = j + ID_last + 1
        # data.ix[j, '分类结果'] = classifier
        # data.ix[j, '详细结果'] = details
        # print 'get clasfier'
        # title = data.ix[j, '标题']
        # bankuai = data.ix[j, '板块']
        # comment = data.ix[j, '内容']
        # sql_time = data.ix[j, '时间']
        # url = data.ix[j, 'url']
        # province = data.ix[j, '省']
        # city = data.ix[j, '市']
        # qu = data.ix[j, '区']
        # come_from = data.ix[j, '网站来源']
        # ITbankuai = data.ix[j, '网站板块']
        # insert_time = data.ix[j, '插入时间']
        id_con = data.ix[j,'Id']
        sql_time = data.ix[j, 'time']
        url = data.ix[j, 'url']
        come_from = data.ix[j, 'comefrom']
        ITbankuai = data.ix[j, 'website']
        insert_time = data.ix[j, 'inserttime']
        # print(title,bankuai,sql_time,url,province,city,qu,come_from,ITbankuai,insert_time)
        # print(data)
        cur = conn.cursor()
        cur.execute(
            "insert into similar_test (ID_relative,title,forum,topic,name,act,content,time,url,P,C,D,province,city,qu,comefrom,website,inserttime,classification,categories,是否农科创,emotion,emotion_score,keywords,source,level) values('%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s');" % (
                id_con,heading, forum,theme,name,act, content, sql_time, url,sheng,shi,district, province, city, qu, come_from, ITbankuai, insert_time,
                classifier, details, nkc,emotion_result,emotion_num, topic_words,var_source,var_level))
        print('insert yet')
        cur.close()
    except:
        print('insert error')


#
# def key_words(content):
#     words = filter(lambda w: w not in stopwords and len(w) > 1, jieba.cut(content))
#     count = Counter(words)
#     print(count)
#     count = sorted(count.items(), key=lambda x: x[1], reverse=True)
#     return count[0:3]


def top3(p):
    top_index = np.zeros(3)
    top_prob = np.zeros(3)
    top3_list = []
    for i in range(3):
        top_index[i] = p.argsort()[-i - 1]
    top_index = top_index.astype(int)
    for j in range(3):
        top_prob[j] = p[top_index[j]]
    # top3_list.append(top_index)
    #    top3_list.append(top_prob)
    return top_index, top_prob


if __name__ in '__main__':
    ssx = pd.read_table('D:\文本分类\\省市区.txt', encoding='utf-8')
    province_ssx = []
    city_ssx = []
    district_ssx = []
    for i in range(len(ssx)):
        province_ssx.append(ssx['省'][i])
        city_ssx.append(ssx['市'][i])
        district_ssx.append(ssx['县'][i])
    filepath = 'D:\文本分类\\addr.txt'
    fb = open(filepath, 'r', encoding='utf8')
    addr = json.load(fb)
    fb.close()
    province_list = []
    city_list = []
    qu_list = []
    for i in range(len(addr)):
        addr_province = addr[i]
        PROVINCE = addr_province['name']
        province_list.append(PROVINCE)
        for j in range(len(addr_province['city'])):
            CITY = addr_province['city'][j]['name']
            QU = addr_province['city'][j]['area']
            city_list.append(CITY)
            qu_list.append(QU)
    emotion_words_bag = pd.read_table('D:\文本分类\文本情感分析\\news_analysis\emotion.txt', encoding='utf-8')
    posdict = []
    negdict = []
    posemotion = []
    negemotion = []
    possentiments = []
    negsentiments = []
    for i in range(len(emotion_words_bag)):
        if emotion_words_bag[emotion_words_bag.columns[6]][i] == 1:
            posdict.append(emotion_words_bag[emotion_words_bag.columns[0]][i])
            posemotion.append(emotion_words_bag[emotion_words_bag.columns[5]][i])
            possentiments.append(emotion_words_bag[emotion_words_bag.columns[4]][i])
        if emotion_words_bag[emotion_words_bag.columns[6]][i] == 2:
            negdict.append(emotion_words_bag[emotion_words_bag.columns[0]][i])
            negemotion.append(emotion_words_bag[emotion_words_bag.columns[5]][i])
            negsentiments.append(emotion_words_bag[emotion_words_bag.columns[4]][i])
    file = codecs.open('D:\文本分类\chinese_stopword.txt', 'r', encoding='utf8')
    stopwords = file.read()
    file.close()
    f1 = open('D:\文本分类\pickle\wordlist.txt', 'rb')
    words_list = pickle.load(f1)
    f1.close()
    f2 = open('D:\文本分类\pickle\\filenamelist.txt', 'rb')
    filename_list = pickle.load(f2)
    f2.close()
    f3 = open('D:\文本分类\pickle\categorylist.txt', 'rb')
    category_list = pickle.load(f3)
    f3.close()
    f4 = open('D:\文本分类\pickle\\feature.txt', 'rb')
    nolabel_feature = pickle.load(f4)
    f4.close()
    #    pre_text = codecs.open('D:\文本分类\\新能源.txt', 'r', encoding='utf-8')
    #    pre_text = pre_text.read()
    category_list_v = np.unique(category_list)
    mostdict_ = codecs.open('D:\文本分类\文本情感分析\\news_analysis\most.txt', 'r', encoding='utf8')
    mostdict = mostdict_.read()
    mostdict_.close()
    verydict_ = codecs.open('D:\文本分类\文本情感分析\\news_analysis\\very.txt', 'r', encoding='utf8')
    verydict = verydict_.read()
    verydict_.close()
    file = codecs.open('D:\文本分类\文本情感分析\\news_analysis\more.txt', 'r', encoding='utf8')
    moredict = file.read()
    file.close()
    ishdict_ = codecs.open('D:\文本分类\文本情感分析\\news_analysis\ish.txt', 'r', encoding='utf8')
    ishdict = ishdict_.read()
    ishdict_.close()
    insufficientdict_ = codecs.open('D:\文本分类\文本情感分析\\news_analysis\insufficiently.txt', 'r', encoding='utf8')
    insufficientdict = insufficientdict_.read()
    insufficientdict_.close()
    inversedict_ = codecs.open('D:\文本分类\文本情感分析\\news_analysis\inverse.txt', 'r', encoding='utf8')
    inversedict = inversedict_.read()
    inversedict_.close()
    print('----------------------->开始测试<------------------------')
    print(category_list_v)
    #    ch2_sx_np = tfidf_mat_pre(pre_text)
    #    clf = joblib.load("D:\文本分类\\train_model.m")
    #    p = clf.predict_proba(ch2_sx_np).flatten()
    #    #print(p)
    #    top_index ,top_prob = top3(p)
    #    #print(top_index,top_prob)
    #    first = top_prob[0]
    #    second = top_prob[1]
    #    third = top_prob[2]
    #    if first < 0.1:
    #        classifier = '无法判别'
    #    else:
    #        if first/second >=2:
    #            classifier = '%s>>概率: %.3f'%(category_list_v[top_index[0]],top_prob[0])
    #        else:
    #            if first/third >=2:
    #                classifier = '%s>>概率: %.3f,%s>>概率: %.3f'%(category_list_v[top_index[0]],top_prob[0],category_list_v[top_index[1]],top_prob[1])
    #            else:
    #                classifier = '%s>>概率: %.3f,%s>>概率: %.3f,%s>>概率: %.3f'%(category_list_v[top_index[0]],top_prob[0],category_list_v[top_index[1]],top_prob[1],category_list_v[top_index[2]],top_prob[2])
    #    print(classifier)
    # content_list = {'beijing_content': 253291, 'chongqing_content': 121872, 'anhui_content': 811403,
    #                 'fujian_content': 4052359,
    #                 'gansu_content': 108981, 'guangdong_content': 360138, 'guangxi_content': 289346,
    #                 'guizhou_content': 925040,
    #                 'guojia_content': 154045, 'hainan_content': 438829, 'hebei_content': 67463,
    #                 'heilongjiang_content': 29831,
    #                 'henan_content': 652373, 'hubei_content': 1198010, 'hunan_content': 1441398,
    #                 'jiangshu_content': 2539959, 'zhejiang_content': 1330659,
    #                 'jiangxi_content': 248255, 'jilin_content': 1210173, 'liaoning_content': 398400,
    #                 'shanxi_content': 832471, 'ningxia_content': 107008,
    #                 'shandong_content': 1364968, 'shanghai_content': 1378145, 'shan_xi_content': 864196,
    #                 'tianjin_content': 89092, 'xinjiang_content': 577004,'sichuan_content':52147,
    #                 'xizang_content': 118616, 'yunnan_content': 1454458,'qinghai_content':98987
    #                 }  # //测试用
    content_list = {'similar':0
                        }  # //测试用
    time1 = time.time()
    error =0
    regex = re.compile('[a-zA-Z]')
    while True:
        for i in content_list:
            time.sleep(0.5)
            conn = pymysql.connect(
                host='x',
                port=3306,
                user='x',
                passwd='x',
                db='most',
                charset='utf8'
            )
            table_name = str(i)
            # print table_name
            ID_last = content_list[i]
            # print ID_last
            # id > % s and id < % s
            try:
                table_name = str(i)
                # print table_name
                ID_last = content_list[i]
                sqlcmd = "SELECT * FROM " + '%s where id>%s and id<%s ' % (table_name, ID_last, ID_last + 1000)
                data = pd.read_sql(sqlcmd, conn)
                print('get data')
                conn.close()
                IDs = data['Id']
                # ID = data.index
                # print(ID.max())
                if ID_last == IDs.max():
                    print('ID_last==ID.max()')
                else:
                    # print ID_last
                    # data=data.drop(['id'], axis=1)
                    contents = data['content']
                    headings = data['title']
                    # IDs = data['Id']
                    sources = data['comefrom']
                    forums = data['forum']  # 板块
                    provinces = data['province']  # 省
                    cities = data['city']  # 市
                    qus = data['qu']  # 区
                    content_list[i] = IDs.max()
                    # print 'new_idlast'+str(content_list[i])
                    print('get contents')
                    conn = pymysql.connect(
                        host='60.191.74.66',
                        port=3306,
                        user='lwj',
                        passwd='123456',
                        db='most',
                        charset='utf8'
                    )
                    for j in range(len(contents)):
                        content = contents[j]
                        heading = headings[j]
                        source = sources[j]
                        forum = forums[j]
                        city = cities[j ]
                        qu = qus[j ]
                        province = provinces[j ]
                        # var_source = source
                        # var_level = ''
                        # flag = 0
                        # print j
                        # print ID_last
                        try:
                            classifier_insertSql(content, heading, source, forum, j)
                        except:
                            print(None)
                            error += 1
                    conn.close()

            except:
                print('waiting for data','错误统计：%s'%(error))
                time2 = time.time()
                print(time2 - time1)
