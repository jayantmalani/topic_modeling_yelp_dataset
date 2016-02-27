
# coding: utf-8

# In[5]:

from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pickle
import nltk #a python natual language toolkit. Has pruning resources
import pandas as pd


# In[6]:

# Data loaded 
pitts_bizs = pickle.load(open('pittsburgh_bizs.p','rb'))


# These are all bussinesses. First we should filter out all but the resteurants and other food vendors, to hopefully limit the number of topics to changes in taste, rather than different vocabularies for different bussinesses. We can use the Yelp categories.json file to do that.
# 
# Then, let's first try doing a topic model for all the reviews from the year 2015. Before we throw the corpus into gensim, let's try and use nltk's stemmer to simplify the text by removing unneeded conjugations, pluralizations.

# In[7]:

#Data modified to bizframe format
bizframe = pd.DataFrame(pitts_bizs)


# In[60]:

reviewList = []
for i in range(len(bizframe['reviews'])):
    for j in range(len(bizframe['reviews'][i])):
        reviewString = bizframe['reviews'][i][j]['text']
        categoryList = bizframe['categories'][i]
        for category in categoryList:
               if category in ['Food','Restaurants','Hotels']:
                    reviewList.append(reviewString)
                    break


# In[61]:

# This cell clears the stop words and also performs Stemming operation to reduce topically similar words in their root
# Output is stored in texts
# Simple regex Tokenizer
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list from stop-word package
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# list for tokenized documents in loop
texts = []

# loop through document list
for i in reviewList:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)


# In[ ]:


# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]


# In[35]:


# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=2, num_words=4))

