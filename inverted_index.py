import re


def token_stream(line):                            #返回list.line
    return re.findall(r'\w+', line, flags=re.I)    #re.I 忽略大小写,去掉标点符号

#list = ['a','v','好','kan']
#lineNum = 80
def mapper(lineNum, list):                         #构建mapper字典 已存在的append 1 不存在的初始化1
    dic = {}                                       #{'80:a': [1, 1], '80:好': [1], '80:v': [1], '80:kan': [1]}
    for item in list:
        key = ''.join([str(lineNum), ':', item])
        if key in dic:
            ll = dic.get(key)
            ll.append(1)
            dic[key] = ll
        else:
            dic[key] = [1]

    return dic


def reducer(dic):                                  #{'a': ['80:[1, 1]'], '好': ['80:[1]'], 'kan': ['80:[1]'], 'v': ['80:[1]']}

    keys = dic.keys()
    rdic = {}
    for key in keys:
        lineNum, kk = key.split(":")
        ss = ''.join([lineNum, ':', str(dic.get(key))])
        if kk in rdic:
            ll = rdic[kk]
            ll.append(ss)
            rdic[kk] = ll
        else:
            rdic[kk] = [ss]

    return rdic


def combiner(dic):                                 #{'80:a': 2, '80:好': 1, '80:v': 1, '80:kan': 1}

    keys = dic.keys()
    tdic = {}
    for key in keys:
        valuelist = dic.get(key)
        count = 0
        for i in valuelist:
            count += i
        tdic[key] = count
    return tdic


def shuffle(dic):                                 #列表[('80:a', [1, 1]), ('80:kan', [1]), ('80:v', [1]), ('80:好', [1])]
    dict = sorted(dic.items(), key=lambda x: x[0])
    return dict


def get_reverse_index(filepath):
    file = open(filepath, 'r',encoding='utf8')
    lineNum = 0
    rdic_p = {}
    while True:
        lineNum += 1
        line = file.readline()
        if line != '':
            # print lineNum, ' ', line, ;
            pass
        else:
            break                                 #读到出现空值
        list = token_stream(line)
        mdic = mapper(lineNum, list)
        cdic = combiner(mdic)
        # print cdic
        rdic_p.update(cdic)

    rdic = reducer(rdic_p)

    #sdic = shuffle(rdic)
    return rdic


if __name__ == '__main__':

    filepath = input("输入文件目录 ：");
    dic = get_reverse_index(filepath)

    search_word = input("输入查询词 ：");

    if search_word in dic:
        print (dic.get(search_word))
    else:
        print (-1)
