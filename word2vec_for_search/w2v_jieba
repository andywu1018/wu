# -*- coding: utf-8 -*-
# author@datassis.com
import jieba
import jieba.posseg as  pseg
import jieba.analyse
import codecs,sys

def readtext(path):
    text = codecs.open(path, 'r', encoding='utf-8')
    content = text.read()
    text.close()
    return content


if __name__ == '__main__':
    stopwords = readtext('D:\文本分类\chinese_stopword.txt')
    f= codecs.open('D:/搜索/wiki.zh.simp.txt','r',encoding='utf8')
    target = codecs.open('D:/搜索/wiki.zh.simp.seg.txt','w',encoding='utf8')
    print('open files')

    lineNum = 1
    line = f.readline()
    while line:
        print('-----processing----',lineNum,'articles----')
        seg_list =jieba.cut(line,cut_all=False)

        line_seg = ' '.join(seg_list)
        target.writelines(line_seg)
        lineNum = lineNum+1
        line = f.readline()

    print('done')
    f.close()
    target.close()
