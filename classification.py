#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_style("darkgrid")

import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import gensim
from gensim.utils import simple_preprocess
from gensim.models import phrases, word2vec, Word2Vec
from gensim.models.phrases import Phrases, Phraser

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[77]:


df = pd.read_csv('/Users/13121510983163.com/Desktop/Product-Categorization-NLP-master/data/products_final.csv', header=0,index_col=0)
df.head()


# In[78]:


text = df.drop(['product_type', 'currency', 'id', 'price', 'price_sign', 'rating', 'brand', 'category'], axis=1)


# In[79]:


text.head()


# In[80]:


text['description'] = text['description'].astype(str)
text['tag_list'] = text['tag_list'].astype(str)
text['name'] = text['name'].astype(str)


# In[81]:


# This function converts a text to a sequence of words:
def tokens(words):
    words = re.sub("[^a-zA-Z]"," ",words)
    text = words.lower().split()
    return " ".join(text)


# In[82]:


text['description'] = text['description'].apply(tokens)
text['tag_list'] = text['tag_list'].apply(tokens)
text['name'] = text['name'].apply(tokens)
text.head()


# In[83]:


stop = stopwords.words('english')
stop[0:10]


# In[84]:


def stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)


# In[85]:


text['description'] = text['description'].apply(stopwords)
text['name'] = text['name'].apply(stopwords)
text.head()


# In[99]:


lem = WordNetLemmatizer()

def word_lem(text):
    lem_text = [lem.lemmatize(word) for word in text.split()]
    return " ".join(lem_text)


# ## Model creation

# ### Word2vec model for description

# In[100]:


model = Word2Vec(sentences, min_count=1, vector_size=100, window=3)


# In[101]:


#This will print the most similar words present in the model: 
res1=model.wv.most_similar('liner')
res2=model.wv.most_similar('eyeshadow')
res3=model.wv.most_similar('mascara')


# In[102]:


dict1=dict(res1)
dict2=dict(res2)
dict3=dict(res3)
list1=list(dict1.keys())
list2=list(dict2.keys())
list3=list(dict3.keys())
print('liner:', list1, '\n', 'eyeshadow:', list2, '\n', 'mascara:', list3)


# In[103]:


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
    return translate_results; 


# In[104]:


tra1 = fy(list1)
tra2 = fy(list2)
tra3 = fy(list3)
print(' 眼线笔:', tra1, '\n', '眼影:', tra2, '\n', '睫毛膏:', tra3)

