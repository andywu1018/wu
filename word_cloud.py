import matplotlib.pyplot as plt
from  os import  path
from scipy.misc import imread
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import jieba

imagepath = 'C:\\Users\Administrator\Desktop\\kobe1.jpg'
text_from_file_with_apath = open('C:\\Users\Administrator\Desktop\kobe.txt',encoding='utf8').read()
kobe_coloring = imread(path.join(imagepath))

wordlist_after_jieba = jieba.cut(text_from_file_with_apath, cut_all=False)
wl_space_split = " ".join(wordlist_after_jieba)

my_wordcloud = WordCloud(font_path='C:/Windows/Fonts/STXINGKA.TTF',mask=kobe_coloring,background_color='grey').generate(wl_space_split)
image_colors = ImageColorGenerator(kobe_coloring)

# plt.imshow(my_wordcloud)
# plt.axis("off")
# plt.figure()
# recolor wordcloud and show
# we could also give color_func=image_colors directly in the constructor
plt.imshow(my_wordcloud.recolor(color_func=image_colors))
plt.axis("off")
plt.figure()
plt.imshow(kobe_coloring, cmap=plt.cm.hot)
plt.axis("off")
plt.show()
