# -*- coding: utf-8 -*-
# author@datassis.com
import warnings
warnings.filterwarnings(action='ignore',category=UserWarning,module='gensim')

import logging
import os.path
import sys
import multiprocessing

from gensim.corpora import WikiCorpus
from gensim.models import  Word2Vec
from gensim.models.word2vec import LineSentence


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level = logging.INFO)
    logging.info('running %s' % ' '.join(sys.argv))

    fdir = 'D:/搜索/'
    inp =  fdir+'wiki.zh.simp.seg.txt'
    outp1 = fdir+'wiki.zh.text.model'
    outp2 = fdir+'wiki.zh.text.vector'

    model  = Word2Vec(LineSentence(inp),size=400,window=5,min_count=5,workers=multiprocessing.cpu_count())

    model.save(outp1)
    model.wv.save_word2vec_format(outp2,binary=False)
