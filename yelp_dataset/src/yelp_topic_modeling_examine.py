
# coding: utf-8

# In[6]:

from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pickle
import nltk #a python natual language toolkit. Has pruning resources
import pandas as pd
import operator
import unicodedata

import itertools
import numpy as np

from multiprocessing import Pool


# In[7]:

def output_topic_model(ldamodel,output_fname='',num_words=10):
    ''' For outputting the results of a topic model 
    into a more human-readable format
    '''
    
    #Format final output
    mat = ldamodel.print_topics(num_topics=10, num_words=num_words)
    final_output = []
    
    for i in mat:
        topic = i[1]
        weights_words = topic.split('+')
        words = [w.split('*')[1] for w in weights_words] #Get rid of weights
        words = [unicodedata.normalize('NFKD', word).encode('ascii','ignore') for word in words]
        final_output.append(words)
    
    output_frame = pd.DataFrame(final_output)

    # If a filename has been specified, save the topic model
    if output_fname:
        output_frame.to_csv(output_fname,encoding='utf-8')
   
    return output_frame


# In[8]:

# Load the results of topic modelling with additional computational passes
bigram_mods = pickle.load(open('tms_bigrams_mp.p','rb'))
unigram_mods = pickle.load(open('tms_unigrams_mp.p','rb'))


# In[ ]:

#Uncomment this when the _all_ topic models, using the 2010-2015 corpuses, have finished generating:

"""
unigram_all = pickle.load(open('tms_all_unigram_mp.p','rb'))
bigram_all = pickle.load(open('tms_all_bigram_mp.p','rb'))

#Gentrified and ungentrified bigram topic model representations
g_all = output_topic_model(mod_gent,'tm_all_mp_bigram_10w_gent.csv',num_words)
ung_all = output_topic_model(mod_ungent,'tm_all_mp_bigram_10w_ungent.csv',num_words)

#Gentrified and ungentrified unigram topic model representations
g_all_u = output_topic_model(mod_gent,'tm_all_mp_unigram_10w_gent.csv',num_words)
ung_all_u = output_topic_model(mod_ungent,'tm_all_mp_unigram_10w_ungent.csv',num_words)

g_all
"""


# In[14]:

#Retrieve the topic models themselves from the pickle dumps

#bigram topic models
mod_g2015 = bigram_mods[0] # bigram topic model for 2015, gentrified-area reviews
mod_g2010 = bigram_mods[1] 
mod_u2015 = bigram_mods[2]
mod_u2010 = bigram_mods[3]

#unigrams topic models
mod_g2015u = unigram_mods[0] # unigram topic model for 2015, gentrified-area reviews
mod_g2010u = unigram_mods[1]
mod_u2015u = unigram_mods[2]
mod_u2010u = unigram_mods[3]

#Get human-readable representations of the topic models as panda dataframes
#When not given a filename, the function will not write out a file
num_words = 10

#bigrams
g2015 = output_topic_model(mod_g2015,num_words=num_words) # Pandas dataframe representation of topic model
g2010 = output_topic_model(mod_g2010,num_words=num_words)
u2015 = output_topic_model(mod_u2015,num_words=num_words)
u2010 = output_topic_model(mod_u2010,num_words=num_words)

#unigrams
g2015u = output_topic_model(mod_g2015u,num_words=num_words) 
g2010u = output_topic_model(mod_g2010u,num_words=num_words)
u2015u = output_topic_model(mod_u2015u,num_words=num_words)
u2010u = output_topic_model(mod_u2010u,num_words=num_words)

# Throw away the individual topics, get unordered lists of the 100 most weighted words
g15 = set(g2015.values.flatten()) # all 100 of the most weighted words for gentrified, 2015
g10 = set(g2010.values.flatten())
u15 = set(u2015.values.flatten())
u10 = set(u2010.values.flatten())
g15u = set(g2015u.values.flatten())
g10u = set(g2010u.values.flatten())
u15u = set(u2015u.values.flatten())
u10u = set(u2010u.values.flatten())


# In[15]:

g2015


# In[23]:

#Make some difference lists, easily formatted for copy-pasting
print ', '.join(g15.difference(u15))
print ', '.join(g10.difference(u10))


# In[ ]:



