#!/usr/bin/env python
# coding: utf-8
from gettext import install

import matplotlib
import matplotlib_inline
import pandas as pd
from IPython import get_ipython
from nbformat.v1 import upgrade

if not get_ipython() == None:
    assert isinstance(get_ipython().run_line_magic, matplotlib, matplotlib_inline)

import seaborn as sns

sns.set_style("darkgrid")

import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec

df = pd.read_csv('/Users/13121510983163.com/Desktop/KOL/Product-Categorization-NLP-master/data/products_final.csv',
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
print('liner:', list1, '\n', 'eyeshadow:', list2, '\n', 'mascara:', list3)

from urllib import request, parse
import json


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


tra1 = fy(list1)
tra2 = fy(list2)
tra3 = fy(list3)
print(' 眼线笔:', tra1, '\n', '眼影:', tra2, '\n', '睫毛膏:', tra3)
