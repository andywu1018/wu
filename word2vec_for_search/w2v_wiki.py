# -*- coding: utf-8 -*-
# author@datassis.com
import logging
import os.path
import sys

from gensim.corpora import WikiCorpus

if __name__ == '__main__':
    program =os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logging.info('running %s' % ' '.join(sys.argv))

    if len(sys.argv) < 3 :
        print(globals()['__doc__']%locals())
        sys.exit(1)

    inp , outp = sys.argv[1:3]
    space = " "
    i= 0

    output = open(outp,'w',encoding='utf8')
    wiki = WikiCorpus(inp,lemmatize=False,dictionary={})
    for text in wiki.get_texts():
        output.write(space.join(text)+"\n")
        i = i+1
        if (i % 10000 == 0 ):
            logger.info("Saved" + str(i)+ "articles.")

    output.close()
    logger.info("FINISHED"+str(i)+'ARTICLES.')


 # D:\>python D:/搜索/word2vec_wiki.py zhwiki-latest-pages-articles.xml.bz2 wiki.zh.txt
 #
