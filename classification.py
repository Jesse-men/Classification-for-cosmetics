#!/usr/bin/env python
# coding: utf-8
from gettext import install
import matplotlib
import matplotlib_inline
import pandas as pd
from IPython import get_ipython
from nbformat.v1 import upgrade
import seaborn as sns
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Word2Vec
from urllib import request, parse
import json

if not get_ipython() == None:
    assert isinstance(get_ipython().run_line_magic, matplotlib, matplotlib_inline)

sns.set_style("darkgrid")


df = pd.read_csv('../products_final.csv',
                 header=0, index_col=0)
text = df.drop(['product_type', 'currency', 'id', 'price', 'price_sign', 'rating', 'brand', 'category'], axis=1)

text['description'] = text['description'].astype(str)
text['tag_list'] = text['tag_list'].astype(str)
text['name'] = text['name'].astype(str)


# This function converts a text to a sequence of words:
def tokens(words):
    words = re.sub("[^a-zA-Z]", " ", words)
    text = words.lower().split()
    return " ".join(text)


text['description'] = text['description'].apply(tokens)
text['tag_list'] = text['tag_list'].apply(tokens)
text['name'] = text['name'].apply(tokens)


stop = stopwords.words('english')
stop[0:10]


def stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)


text['description'] = text['description'].apply(stopwords)
text['name'] = text['name'].apply(stopwords)

lem = WordNetLemmatizer()


def word_lem(text):
    lem_text = [lem.lemmatize(word) for word in text.split()]
    return " ".join(lem_text)


# ## Model creation
# ### Word2vec model for description
sentences = [row.split() for row in text['description']]
model = Word2Vec(sentences, min_count=1, vector_size=100, window=3)

# This will print the most similar words present in the model:
res1 = model.wv.most_similar('liner')
res2 = model.wv.most_similar('eyeshadow')
res3 = model.wv.most_similar('mascara')

dict1 = dict(res1)
dict2 = dict(res2)
dict3 = dict(res3)
list1 = list(dict1.keys())
list2 = list(dict2.keys())
list3 = list(dict3.keys())


def fy(i):
    req_url = 'http://fanyi.youdao.com/translate'
    Form_Date = {}
    Form_Date['i'] = i
    Form_Date['doctype'] = 'json'
    Form_Date['form'] = 'AUTO'
    Form_Date['to'] = 'AUTO'
    Form_Date['smartresult'] = 'dict'
    Form_Date['client'] = 'fanyideskweb'
    Form_Date['salt'] = '1526995097962'
    Form_Date['sign'] = '8e4c4765b52229e1f3ad2e633af89c76'
    Form_Date['version'] = '2.1'
    Form_Date['keyform'] = 'fanyi.web'
    Form_Date['action'] = 'FY_BY_REALTIME'
    Form_Date['typoResult'] = 'false'

    data = parse.urlencode(Form_Date).encode('utf-8')
    response = request.urlopen(req_url, data)
    html = response.read().decode('utf-8')

    translate_results = json.loads(html)  # 以json格式载入
    translate_results = translate_results['translateResult'][0][0]['tgt']  # json格式调取
    return translate_results


names = [row.split() for row in text['name']]
list_1 = [str(x) for item in names for x in item]
list_2 = tuple(list_1)


def build_dict(list_2):
    word_freq_dict = dict()
    for word in list_2:
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] += 1

    word_freq_dict = sorted(word_freq_dict.items(), key = lambda x:x[1], reverse = True)

    word2id_dict = dict()
    word2id_freq = dict()
    id2word_dict = dict()

    for word, freq in word_freq_dict:
        curr_id = len(word2id_dict)
        word2id_dict[word] = curr_id
        word2id_freq[word2id_dict[word]] = freq
        id2word_dict[curr_id] = word

    return word2id_freq, word2id_dict, id2word_dict


word2id_freq, word2id_dict, id2word_dict = build_dict(list_2)
vocab_size = len(word2id_freq)
for _, (word, word_id) in zip(range(500), word2id_dict.items()):
    a = "睫毛膏%s, 点击率 %d" % ('mascara', word2id_freq[1])
    b = "眼线笔%s, 点击率 %d" % ('eyeliner', word2id_freq[5])
    c = "眼影%s, 点击率 %d" % ('eyeshadow', word2id_freq[65])

tra1 = fy(list1)
tra2 = fy(list2)
tra3 = fy(list3)

d3 = {'睫毛膏': tra1, '眼线笔': tra2, '眼影': tra3}
print('order:', a, b, c, '\n', 'keywords:', d3)
