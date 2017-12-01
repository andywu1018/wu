import pymysql
import json
import math
import time
import collections
import re
import numpy as np
import pandas as pd

re_chinese = re.compile(u'[^\u4e00-\u9fa5]+')
def readText(corpus):
    corpuses = re_chinese.sub('', corpus.rstrip())  # 使用list保存string，然后用join方法来合并，效率比"+"更高
    reverse_corpus = corpuses[::-1]
    return corpuses, reverse_corpus

def read_list(path):
    fp = open(path, 'r', encoding='utf8')
    lines = []
    for line in fp.readlines():
        line = line.strip()
        lines.append(line)
    fp.close()
    return lines
stopwords = read_list('C:\\Users\Administrator\Desktop\stopwords.txt')


def dictionary(corpus, reverse_corpus):  # w：词语最大长度，建立备选词典
    sen_len = int(len(corpus))  # 语料长度
    for j in range(1, 9):
        for i in range(sen_len - j, -1, -1):
            if corpus[i:i + j] != '':
                if corpus[i:i + j] not in wordDic:
                    wordDic[corpus[i:i + j]] = 1
                else:
                    wordDic[corpus[i:i + j]] += 1

    sen_len = int(len(reverse_corpus))  # 逆序语料长度
    for j in range(1, 9):
        for i in range(sen_len - j, -1, -1):
            if reverse_corpus[i:i + j] != '':
                if reverse_corpus[i:i + j] not in reverse_wordDic:
                    reverse_wordDic[reverse_corpus[i:i + j]] = 1
                else:
                    reverse_wordDic[reverse_corpus[i:i + j]] += 1


    return wordDic, reverse_wordDic


def entropy(wordDic, reverse_wordDic):
    # 先计算右邻字信息熵
    sorted_WordDic = sorted(wordDic)  # 右邻字信息熵
    i = 0
    length_wordDic = len(sorted_WordDic)
    while (i < length_wordDic):
        word = sorted_WordDic[i]  # 目标词语
        j = i + 1
        if (j >= length_wordDic):
            break
        buffer = {}  # 右邻字及个数
        while (word in sorted_WordDic[j]):
            label = sorted_WordDic[j][len(word)]
            if label not in buffer:
                buffer[label] = wordDic[sorted_WordDic[j]]  # 不要重复加相关右邻字
            j += 1
            if (j >= length_wordDic):
                break
        sum = 0.000000001
        pro_buffer = {}
        for buff in buffer:
            sum = sum + buffer[buff]
        for buff in buffer:
            pro_buffer[buff] = buffer[buff] / sum
        right_entropy = 0.0
        for pro in pro_buffer:
            right_entropy = right_entropy - pro_buffer[pro] * math.log(pro_buffer[pro])
        right_word_entropy[word] = right_entropy  # 保存右邻字信息熵信息
        i += 1

    # 再计算左邻字信息熵
    left_sorted_WordDic = sorted(reverse_wordDic)  # 左邻字信息熵
    i = 0
    length_wordDic = len(left_sorted_WordDic)
    while (i < length_wordDic):
        word = left_sorted_WordDic[i]  # 目标词语
        j = i + 1
        if (j >= length_wordDic):
            break
        buffer = {}  # 左邻字个数
        while (word in left_sorted_WordDic[j]):
            label = left_sorted_WordDic[j][len(word)]
            if label not in buffer:
                buffer[label] = reverse_wordDic[left_sorted_WordDic[j]]  # 不要重复加相关左邻字
            j += 1
            if (j >= length_wordDic):
                break
        sum = 0.000000001
        pro_buffer = {}
        for buff in buffer:
            sum = sum + buffer[buff]
        for buff in buffer:
            pro_buffer[buff] = buffer[buff] / sum
        left_entropy = 0.0
        for pro in pro_buffer:
            left_entropy = left_entropy - pro_buffer[pro] * math.log(pro_buffer[pro])
        words = word[::-1]
        left_word_entropy[words] = left_entropy  # 保存左邻字信息熵信息
        i += 1
    return right_word_entropy, left_word_entropy


def concreation(wordDic):  # 计算完整词语凝固程度
    sum = 0.0
    for i in wordDic:
        sum = sum + wordDic[i]
    for i in wordDic:
        probability_word[i] = wordDic[i] / sum
    for i in wordDic:
        length = len(i)
        if length > 1:
            j = 1
            p = 9999999999
            while (j < length):
                right = i[0:j]
                left = i[j:length]
                k = probability_word[i] / (probability_word[right] * probability_word[left])
                if (p > k):
                    p = k
                j += 1
            concreation_word[i] = p
    return concreation_word


def one2ten(dict):
    one = int(len(dict)/10)
    lst = sorted(list(dict.values()),reverse=True)
    keyandvalue = sorted(dict.items(), key=lambda item: item[1], reverse=True)
    _array = np.array(lst)
    _array[:one] = 10
    _array[one:one*2] = 9
    _array[one*2:one*3] = 8
    _array[one*3:one*4] = 7
    _array[one*4:one*5] = 6
    _array[one*5:one*6] = 5
    _array[one*6:one*7] = 4
    _array[one*7:one*8] = 3
    _array[one*8:one*9] = 2
    _array[one*9:] = 1
    new_lst = _array.tolist()
    one2ten_dict = collections.OrderedDict()
    for i in range(len(keyandvalue)):
        one2ten_dict[keyandvalue[i][0]] = new_lst[i]
    return one2ten_dict



def word_generation(left_entropy, right_entropy, concreation):
    for word in wordDic:
        if word.isdigit() == False:
            if ((len(word)) > 1 and (word in left_word_entropy) and (word in right_word_entropy) and (
                        word in concreation_word)):
                if ((left_word_entropy[word] >= left_entropy) and (right_word_entropy[word] >= right_entropy) and (
                            concreation_word[word] >= concreation)):
                    # print(word)
                    score[word] = concreation_word[word] / left_word_entropy[word] / right_word_entropy[word]
    score_list = one2ten(score)
    return score_list


def wordsfind(content):
    corpus, reverse_corpus = readText(content)
    print('==========开始构建词库:==========' )
    wordDic, reverse_wordDic = dictionary(corpus, reverse_corpus)
    # 最大长度五个字的词语,参数可调
    print('==========开始计算左右信息熵==========')
    right_word_entropy, left_word_entropy = entropy(wordDic, reverse_wordDic)  # 计算左右信息熵
    print('==========开始计算内部凝固得分==========')
    concreation_word = concreation(wordDic)  # 计算内部凝固程度
    print('==========开始计算最终得分==========')
    final_score = word_generation(0.3, 0.3, 500)  # 参数可自行调整
    print('==========开始生成词序列Json=========')
    # for i in range(len(score_list)):
    #     final_score[score_list[i][0]] = score_list[i][1]
    final_score_dic_copy = {}
    freq_dict = {}
    freq = list(map(lambda x:corpus.count(x),list(final_score.keys())))
    for i in range(len(freq)):
        freq_dict[list(final_score.keys())[i]] = freq[i]
    freq_dict = one2ten(freq_dict)
    for word in final_score:
        if len(word) > 2:
            final_score_dic_copy[word] = freq_dict[word]*final_score[word]
    score_list_new = sorted(final_score_dic_copy.items(), key=lambda item: item[1], reverse=True)
    # final_score = collections.OrderedDict()
    newords_score = collections.OrderedDict()
    # for i in range(len(score_list)):
    #     final_score[score_list[i][0]]=score_list[i][1]
    for i in range(len(score_list_new)):
        newords_score[score_list_new[i][0]] = score_list_new[i][1]

    return newords_score


def insertSql(id):
    cur = conn.cursor()
    cur.execute( "insert into gov_tes3 (Id ,keywords) values (%d,%r) ;"%(id,keywords))
    # print('%s insert yet'%(i))
    cur.close()
    # except:

time1 = time.time()
conn = pymysql.connect(
    host='',
    port=3306,
    user='',
    passwd='',
    db='',
    charset='utf8'
)
sqlcmd = "select id,clean_content from public_policy_gov;"
data = pd.read_sql(sqlcmd, conn)
contents = data['clean_content']
IDS = data['id']
conn.close()
conn = pymysql.connect(
    host='',
    port=3306,
    user='',
    passwd='',
    db='',
    charset='utf8'
)
for i in range(len(data)):
    wordDic = {}
    reverse_wordDic = {}  # 反序语料，计算左邻字信息熵
    right_word_entropy = {}
    left_word_entropy = {}
    probability_word = {}
    concreation_word = {}
    score = {}
    final_score = {}
    content = contents[i]
    id = IDS[i]
    words = wordsfind(content)
    keywords_lst = []
    count = 0
    for word in list(words.keys()):
        if count <=4:
            if word not in stopwords:
                keywords_lst.append(word)
                count += 1
        else:
            break
    keywords = ','.join(keywords_lst)
    insertSql(id)
    print('%s ok'%(i))

print('done')

conn.close()
