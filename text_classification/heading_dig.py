# import docx
#
# def readDocx(docName):
#     fullText = []
#     doc = docx.Document(docName)
#     paras = doc.paragraphs
#     for p in paras:
#         fullText.append(p.text)
#     return '\n'.join(fullText)
#
# readDocx('C:\\Users\Administrator\Desktop\周报@数据工程部 - 吴迪 8.11.docx')
#
#nr:人名  ns 地点
#
import pymysql
import time
import pandas as pd
import numpy as np
import re
import jieba
import jieba.posseg

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
        theme4 = ','.join(np.unique(re.findall('对(.*?)进行调研', heading)))
        theme5 = ','.join(np.unique(re.findall('指导(.*?)建设', heading)))
        theme6 = ','.join(np.unique(re.findall('调研(.*?)建设', heading)))
        theme7 = ','.join(np.unique(re.findall('指导(.*?)工作', heading)))
        theme8 = ','.join(np.unique(re.findall('视察(.*?)工作', heading)))
        theme9 = ','.join(np.unique(re.findall('考察(.*?)情况', heading)))
        theme10 = ','.join(np.unique(re.findall('推进(.*?)建设', heading)))
        listtheme = [theme1,theme2,theme3,theme4,theme5,theme6,theme7,theme8,theme9,theme10]
        while '' in listtheme:
            listtheme.remove('')
        theme = ','.join(listtheme)
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
        # print(spilit_dict)
        print("地点：省：\033[1;34m %s \033[0m市：\033[1;36m %s \033[0m区：\033[1;31m %s \033[0m人物:\033[1;35m %s \033[0m行为:\033[1;32m %s \033[0m主题:\033[1;36m %s \033[0m"%(sheng,shi,qu,name,act,theme))
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    return spilit_dict


    # content_list = {'zixun_content':0
    #                     }  # //测试用
if __name__ in '__main__':
    ssx = pd.read_table('D:\文本分类\\省市区.txt', encoding='utf-8')
    province_ssx = []
    city_ssx  = []
    district_ssx = []
    for i in range(len(ssx)):
        province_ssx.append(ssx['省'][i])
        city_ssx.append(ssx['市'][i])
        district_ssx.append(ssx['县'][i])
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
    #                 'tianjin_content': 89092, 'xinjiang_content': 577004, 'sichuan_content': 52147,
    #                 'xizang_content': 118616, 'yunnan_content': 1454458, 'qinghai_content': 98987
    #                 }  # //测试用
    content_list = {'similar':0
                        }  # //测试用
    time1 = time.time()
    error =0
    while True:
        for i in content_list:
            time.sleep(0.5)
            conn = pymysql.connect(
                host='60.191.74.66',
                port=3306,
                user='lwj',
                passwd='123456',
                db='most',
                charset='utf8'
            )
            # table_name = str(i)
            # # print table_name
            # ID_last = content_list[i]
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
                    # sources = data['comefrom']
                    forums = data['forum']  # 板块
                    # provinces = data['省']  # 省
                    # cities = data['市']  # 市
                    # qus = data['区']  # 区
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
                        # source = sources[j + ID_last + 1]
                        forum = forums[j]
                        # city = cities[j + ID_last + 1]
                        # qu = qus[j + ID_last + 1]
                        # province = provinces[j + ID_last + 1]
                        # var_source = source
                        # var_level = ''
                        # flag = 0
                        # print j
                        # print ID_last
                        try:
                            spilit_heading(heading, forum)

                        except:
                            print(None)
                            error += 1
                    conn.close()
            except:
                print('waiting for data','错误统计：%s'%(error))
                time2 = time.time()
                print(time2 - time1)
