# -*- coding: utf-8 -*-
# author@datassis.com
import jieba
import jieba.posseg as  pseg
import jieba.analyse
import codecs,sys
import re

def read_list(path):
    fp = open(path, 'r', encoding='utf8')
    lines = []
    for line in fp.readlines():
        line = line.strip()
        lines.append(line)
    fp.close()
    return lines
stopwords = read_list('D:\\2018年工作\PyProgram\stopwords.txt')


if __name__ == '__main__':
    f= codecs.open('D:\\2018年工作\PyProgram/PATENT.txt','r',encoding='utf8')
    target = codecs.open('D:\\2018年工作\PyProgram\mc.zh.simp.seg_total.txt','w',encoding='utf8')
    print('open files')

    lineNum = 1
    line = f.readline()
    re_chinese = re.compile(u'[\u4e00-\u9fa5]+')
    while lineNum!='':
        line = ''.join(re.findall(re_chinese, line)) + '\r\n'
        print('-----processing----',lineNum,'articles----')
        seg_list =[]
        for seg in jieba.cut(line, cut_all=False):
            if len(seg)>=2:
              if seg not in stopwords:
                if seg != ' ':
                      seg_list.append(seg)

        line_seg = ' '.join(seg_list)
        target.writelines(line_seg)
        lineNum = lineNum+1
        line = f.readline()

    print('done')
    f.close()
    target.close()