# -*- coding=utf-8 -*-
# 利用jaccard similarity 计算文本相似度


import os
import time
import numpy as np
import progressbar
import pandas as pd
import pymysql

#############################################
#
# 读取文件，保存到一个字符串中
# 输入： 文件名完整路径
# 输出： 文件内容
#
#############################################
def readFile(file_name):
    f = open(file_name, "r",encoding='utf8')
    file_contents = f.read()
    file_contents = file_contents.replace("\t", "")
    file_contents = file_contents.replace("\r", "")
    file_contents = file_contents.replace("\n", "")
    f.close()
    return file_contents  # to_unicode_or_bust(file_contents)


#############################################
#
# 分割字符串，使用k-shingle方式进行分割
# 输入：字符串，k值
# 输出：分割好的字符串，存入数组中
#
#############################################
def splitContents(content, k=5):
    content_split = []
    for i in range(len(content) - k):
        content_split.append(content[i:i + k])
    return content_split


#############################################
#
# 将数据保存到hash表中，也就是某个集合
# 输入：已经分隔好的数据
# 输出：hash表
#
#############################################
def hashContentsList(content_list,Id):
    hash_content = {}
    for i in content_list:
        if i in hash_content:
            hash_content[i] = hash_content[i] + 1
        else:
            hash_content[i] = 1
    hash_content['ID']=Id
    return hash_content

#############################################
#
# 计算交集
# 输入：两个hash表
# 输出：交集的整数
#
#############################################
def calcIntersection(hash_a, hash_b):
    intersection = 0
    if (len(hash_a) <= len(hash_b)):
        hash_min = hash_a
        hash_max = hash_b
    else:
        hash_min = hash_b
        hash_max = hash_a

    for key in hash_min :
        if key not in ['ID']:
            if key in hash_max:
                if (hash_min[key] <= hash_max[key]):
                    intersection = intersection + hash_min[key]
                else:
                    intersection = intersection + hash_max[key]
    return intersection


#############################################
#
# 计算并集
# 输入：两个hash表
# 输出：并集的整数
#
#############################################
def calcUnionSet(hash_a, hash_b, intersection):
    union_set = 0

    for key in hash_a:
        if key not in ['ID']:
          union_set = union_set + hash_a[key]
    for key in hash_b:
        if key not in ['ID']:
          union_set = union_set + hash_b[key]

    return union_set - intersection


#############################################
#
# 计算相似度
# 输入：交集和并集
# 输出：相似度
#
#############################################
def calcSimilarity(intersection, union_set):
    if (union_set > 0):
        return float(intersection) / float(union_set)
    else:
        return 0.0


#############################################
#
# 从某个文本文件获取一个集合，该集合保存了文本中单词的出现频率
# 输入：文件名，k值,默认为5
# 输出：一个词频的hash表
#
#############################################
def getHashInfoFromFile(content,Id,k=5):
    # content = readFile(file_name)
    content_list = splitContents(content, k)
    hash_content= hashContentsList(content_list,Id)
    return hash_content

#############################################
#
# 计算两两相似度
# 输入：哈希数据列表
# 输出：相似度数组
#
#############################################
def calcEachSimilar(hash_contents):
    print(u"计算所有文本互相之间的相似度....")
    start = time.time()
    l = len(hash_contents)
    all = float(len(hash_contents))
    pos = 0.0
    pro = progressbar.ProgressBar().start()
    similar_id = []
    for v1 in (hash_contents):
        similar_list = []
        target_list = []
        pos = pos + 1
        rate_num = int(pos / all * 100)
        pro.update(rate_num)
        # time.sleep(0.1)
        # print "%02d" % int(pos/all*100),
        for v2 in (hash_contents):
            if (v1 != v2 ):
                intersection = calcIntersection(v1, v2)  # 计算交集
                union_set = calcUnionSet(v1, v2, intersection)  # 计算并集
                similar = calcSimilarity(intersection, union_set)
                similar_list.append([similar,v1['ID'],v2['ID']])
        similar_list.sort()
        similar_list.reverse()
        similar_list = similar_list[0:5]
        for j in range(5):
            target_list.append(similar_list[j][2])
        similar_id.append(target_list)
    print(similar_id)
            # else:
            #     similar = 0
            #     similar_list.append(similar)
                # print v1[1]+ "||||||" + v2[1] + " similarity is : " + str(calcSimilarity(intersection,union_set)) #计算相似度
    pro.finish()
    # similar_array = []
    # for i in range(l):
    #     similar_split = similar_list[i*l:i*l+l]
    #     similar_array.append(similar_split)
    # similar_array = np.array(similar_array)
    # similar_list.sort()
    # similar_list.reverse()
    end = time.time()
    print(u"计算所有文本互相之间的相似度结束，用时: " + str(end - start) + u"秒")
    return similar_list


#############################################
#
# 主程序
# 输入:路径和k-shingle中的k值
# 输出:两两相似度数组
#
#############################################
if __name__ == '__main__':
    conn = pymysql.connect(
                host='60.191.74.66',
                port=3306,
                user='lwj',
                passwd='123456',
                db='zhejiang_zixun',
                charset='utf8'
    )
    table_name = 'zhejiang_content'
    ID = 300
    try:
        sqlcmd = " SELECT * FROM " +'%s WHERE id>%s and id<%s' %(table_name,ID,ID+1000)
        data = pd.read_sql(sqlcmd,conn)
        print('get content')
        conn.close()
        contents = data['内容']
        ids = data['id']
        # conn = pymysql.connect(
        #                host='x',
        #                port=3306,
        #                user='x',
        #                passwd='x',
        #                db='most',
        #                charset='utf8'
        #               )
        hash_contents = []
        for j in range(len(contents)):
                       content = contents[j]
                       Id = ids[j]
                       hash_dict = getHashInfoFromFile(content,Id,k=5)
                       hash_contents.append(hash_dict)
                       # print(hash_contents)
        res = calcEachSimilar(hash_contents)
        print(res)
    except:
        print('none')






