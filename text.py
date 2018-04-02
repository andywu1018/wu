#!/uer/bin/python3
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
fdir ="D:\\2018年工作\PyProgram\Train.txt"


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences



#
def train(x_train, size=400, epoch_num=1):
    model_dm = Doc2Vec(min_count=1, window=3, size=size, sample=1e-3, negative=5, workers=6)
    model_dm.build_vocab(x_train.to_array())
    for epoch in range(10):
        model_dm.train(x_train.sentences_perm(),total_examples=model_dm.corpus_count,epochs=model_dm.iter)
        # for epoch in range(10):
        #     model_dm.train(x_train.sentences_perm())
    # model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save("D:\\2018年工作\PyProgram\D2VMODEL")
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
    model_dm = Doc2Vec.load("D:\\2018年工作\PyProgram\D2VMODEL")
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
    inferred_vector_dm = model_dm.infer_vector(text_raw)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
    time03 = time.time()
    print(time03-time02)

    return sims
import time
if __name__ == '__main__':
    # sources = {fdir: '0'}
    # x_train = LabeledLineSentence(sources)
    # # x_train = get_corpus(fdir)
    # time1 = time.time()
    # model_dm = train(x_train)
    # time2 = time.time()
    # print(time2-time1)

    sims = test()
    #
    #
    for count, sim in sims:
        # sentence = x_train[count]
        # words = ''
        # for word in sentence[0]:
        #     words = words + word + ' '
        print(count,sim)

#..........
# from gensim.models.keyedvectors import
# keyedvectors.