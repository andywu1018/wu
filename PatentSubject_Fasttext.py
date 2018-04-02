#!/uer/bin/python3
#encoding:utf-8
import gensim
import numpy as np
import jieba
from numba import vectorize
import re
import multiprocessing
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
# stop_text = open('stop_list.txt', 'r')
# stop_word = []
# for line in stop_text:
#     stop_word.append(line.strip())
TaggededDocument = gensim.models.doc2vec.TaggedDocument
fdir ="D:\\2018年工作\PyProgram\patent.zh.simp.seg_total.txt"

def get_corpus(fdir):

    with open(fdir, 'r',encoding='utf8') as doc:
        docs = doc.readlines()
    train_docs = []
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        length = len(word_list)
        word_list[length - 1] = word_list[length - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        train_docs.append(document)
    return train_docs



def train(x_train, size=150, epoch_num=1):
    model_dm = Doc2Vec(x_train, min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=4)
    # model_dm.build_vocab(x_train)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save("D:\\2018年工作\PyProgram\model_doc2vec")
    # model_dm.batch_words = 20000
    return model_dm

def read_list(path):
    fp = open(path, 'r', encoding='utf8')
    lines = []
    for line in fp.readlines():
        line = line.strip()
        lines.append(line)
    fp.close()
    return lines
stopwords = read_list('D:\\2018年工作\PyProgram\stopwords.txt')

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

def test():
    time0 = time.time()
    model_dm = Doc2Vec.load("D:\\2018年工作\PyProgram\model_doc2vec")
    time01 = time.time()
    print(time01-time0)
    text_test = input('请输入测试文本:')
    time05 = time.time()
    text_raw = filewordProcess(text_test)
    time02 = time.time()
    print(time02-time05)
    # text_cut = jieba.cut(text_test)
    # text_raw = []
    # for i in list(text_cut):
    #     text_raw.append(i)
    inferred_vector_dm = model_dm.infer_vector(text_raw,steps=20,alpha=0.025)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10000)
    time03 = time.time()
    print(time03-time02)

    return sims
import time
if __name__ == '__main__':
    # x_train = get_corpus(fdir)
    # time1 = time.time()
    # model_dm = train(x_train)
    # time2 = time.time()
    # print(time2-time1)

    sims = test()


    for count, sim in sims:
        # sentence = x_train[count]
        # words = ''
        # for word in sentence[0]:
        #     words = words + word + ' '
        print(count,sim)

#model_dm.similar_by_word('产品',5)
