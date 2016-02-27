
# coding: utf-8

# In[17]:

from nltk.tokenize import RegexpTokenizer
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
from collections import Counter as counter


# In[18]:

#Load the charlotte business list that has each business's census tract
char_bizs = pickle.load(open('charlotte_bizs_census.p','rb'))

#Add the zipcode as an element of each business
for business in char_bizs:
    zip_code = business['full_address'][-5:]
    business['zip_code'] = zip_code
char_bizs.sort(key=operator.itemgetter('review_count'),reverse=True)


# In[19]:

# Define the census tracts for our samples
gent_cts = [38.05,37,36,41,7,8,9,14,18.02]
ungent_cts = [38.06, 38.07,38.08,38.02,32.01,39.02,39.02,40,43.05,43.03,43.04,43.02,42,45,44,54.01,54.03,54.04,51,50,49,48,46,45,42,52,53.06,23,18.01,17.01,21,19.19,17.02,16.07,16.09,19.12,16.06,16.05,16.03,16.08,15.04,15.09,15.10,13,53.01,52,53.06,53.05,53.06,58.07,54.07,54.01]


# In[20]:

revs_g2015 = [] #2015 reviews of businesses in gentrified areas
revs_g2010 = [] #2010 reviews of businesses in gentrified areas
revs_u2015 = [] #2015 reviews of businesses in ungentrified areas
revs_u2010 = [] #2010 reviews of businesses in ungentrified areas

revs_gen_all = [] #all reviews of businesses in gentrified areas, 2010-2015
revs_ungen_all = [] #all reviews of businesses in ungentrified areas, 2010-2015

for biz in char_bizs:
    reviews = biz['reviews']
    ct_num = biz['ct_num'] # census track number
    categoryList = biz['categories']
    
    # Use only businesses that serve food
    if set(categoryList).intersection(['Food','Restaurants','Hotels']):
    
        for rev in reviews:
            year = rev['date'].split('-')[0]
            year = int(year)
            revtext = rev['text']

            # Add review to the appropriate category 
            if (ct_num in gent_cts) and (year == 2015):
                revs_g2015.append(revtext)
            if (ct_num in gent_cts) and (year == 2010):
                revs_g2010.append(revtext)
            if (ct_num in ungent_cts) and (year == 2015):
                revs_u2015.append(revtext)
            if (ct_num in ungent_cts) and (year == 2010):
                revs_u2010.append(revtext)
                
            if (ct_num in gent_cts) and (2010 <= year <= 2015):
                revs_gen_all.append(revtext)
            if (ct_num in ungent_cts) and (2010 <= year <= 2015):
                revs_ungen_all.append(revtext)


# In[5]:

print len(revs_g2010), len(revs_g2015), len(revs_u2010),len(revs_u2015)
print len(revs_gen_all), len(revs_ungen_all)


# In[7]:

num_passes = 20
use_bigrams = True


# In[22]:

def create_corpus(reviewList):
    '''A function to produce an lda topic model, 
    given a list of Yelp review texts
    '''

    # I would make these inputs instead of global variable, but I don't know
    # how to make that work with 'map', below
    
    #Check whether or not they're defined yet:
    if (use_bigrams not in locals()) or (num_passes not in locals()):
        print 'Define these in your environment first'

   
    print 'Use Bigrams:'
    print use_bigrams
    print 'Number of Passes:'
    print num_passes
    
    tokenizer = RegexpTokenizer(r'\w+')

    # create English stop words list
    en_stop = get_stop_words('en')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()
    
    # list for tokenized documents in loop
    texts = []

    en_stop.append(u's')
    en_stop.append(u't')
   
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
    
    texts_final = []
    
    if use_bigrams:
        for i in texts:
            text_temp = []
            for j in range(len(i) - 1):
                #print (i[j])
                text_temp.append(i[j] + '_' + i[j+1])
            texts_final.append(text_temp)
    else:
        texts_final = texts
    
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts_final)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts_final]
    
    return corpus


# In[40]:


tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# list for tokenized documents in loop
texts = []

en_stop.append(u's')
en_stop.append(u't')
   
# loop through document list
for i in revs_gen_all:#revs_ungen_all

    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

     # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(stemmed_tokens)

texts_final = []

for i in texts:
    for j in range(len(i) - 1):
        #print (i[j])
        texts_final.append(i[j] + '_' + i[j+1])
    #texts_final.append(text_temp)
texts_final = counter(texts_final)
   
dictWords = texts_final.most_common()
texts_final = pd.DataFrame(dictWords)
texts_final.to_csv('WordCount_gent.csv',encoding='utf-8')


# In[23]:

def char_ldamodel(reviewList):
    
    corpus = create_corpus(reviewList)

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=num_passes)

    return ldamodel


# In[10]:

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


# In[11]:

# We have four topic models to run; let's run them all at once to save time
def easy_parallize(f, sequence):

    from multiprocessing import Pool
    pool = Pool(processes=4) # number of cores(threads?)
    
    result = pool.map(f, sequence) #a list of all the topic models requested
    
    # Dispose of any empty results (needed only if #processes != len(sequence)?)
    cleaned = [x for x in result if not x is None] 
    cleaned = np.asarray(cleaned)
   
    # not optimal but safe
    pool.close()
    pool.join()
    
    return cleaned


# In[ ]:

# produce the four topic models, one for each list of reviews
#%time results = easy_parallize(char_ldamodel,[revs_g2015,revs_g2010,revs_u2015,revs_u2010])


# In[14]:

#Let's try not using bigrams
#%time monoresults = easy_parallize(char_ldamodel,[revs_g2015,revs_g2010,revs_u2015,revs_u2010])


# In[92]:

#Let's try using many passes, and bigrams
# mp stands for 'many passes'
#%time bigram_mp = easy_parallize(char_ldamodel,[revs_g2015,revs_g2010,revs_u2015,revs_u2010])


# In[95]:

#Let's try using many passes, and unigrams
#%time unigram_mp = easy_parallize(char_ldamodel,[revs_g2015,revs_g2010,revs_u2015,revs_u2010])


# In[ ]:

#Let's using many passes, and unigrams
#%time unigram_mp = easy_parallize(char_ldamodel,[revs_g2015,revs_g2010,revs_u2015,revs_u2010])


# In[ ]:

# Let's generate a topic model for all the 2015 data, and all the 2010 data


# In[19]:


# Save or load the topic model results

#pickle.dump(results,open('tms_bigrams.p','wb'))
#pickle.dump(monoresults,open('tms_monograms.p','wb'))
#pickle.dump(bigram_mp,open('tms_bigrams_mp.p','wb'))
#pickle.dump(unigram_mp,open('tms_unigrams_mp.p','wb'))

bigram_mods = pickle.load(open('tms_bigrams_mp.p','rb'))
unigram_mods = pickle.load(open('tms_unigrams_mp.p','rb'))

# Save the results

mod_g2015 = bigram_mods[0] # unigram topic model for 2015, gentrified-area reviews
mod_g2010 = bigram_mods[1]
mod_u2015 = bigram_mods[2]
mod_u2010 = bigram_mods[3]

mod_g2015u = unigram_mods[0] # unigram topic model for 2015, gentrified-area reviews
mod_g2010u = unigram_mods[1]
mod_u2015u = unigram_mods[2]
mod_u2010u = unigram_mods[3]

#Output results 

num_words = 10
g2015 = output_topic_model(mod_g2015,'tm_mp_bigram_10w_gent_2015.csv',num_words)
g2010 = output_topic_model(mod_g2010,'tm_mp_bigram_10w_gent_2010.csv',num_words)
u2015 = output_topic_model(mod_u2015,'tm__mp_bigram_10w_ungent_2015.csv',num_words)
u2010 = output_topic_model(mod_u2010,'tm_mp_bigram_10w_ungent_2010.csv',num_words)
g2015u = output_topic_model(mod_g2015u,'tm_mp_unigram_10w_gent_2015.csv',num_words)
g2010u = output_topic_model(mod_g2010u,'tm_mp_unigram_10w_gent_2010.csv',num_words)
u2015u = output_topic_model(mod_u2015u,'tm_mp_unigram_10w_ungent_2015.csv',num_words)
u2010u = output_topic_model(mod_u2010u,'tm_mp_unigram_10w_ungent_2010.csv',num_words)


# In[ ]:




# **The above calls were from when this notebook was used to both make topic models and examine them. There will be some issues with the variable names if you try to run all the cells above**

# In[33]:

# Create and output both unigram and bigram topic models,
# using all the reviews from 2010-2015, 
# for gentrified an ungentrified regions

num_passes = 1000
use_bigrams = True
get_ipython().magic(u'time bigram_all = easy_parallize(char_ldamodel,[revs_gen_all,revs_ungen_all])')
pickle.dump(bigram_all,open('tms_all_bigrams_mp.p','wb'))
#We write out the entire collection of topic models

num_passes = 1000
use_bigrams = False #passing so-called inputs to functions like this is bad form, but 
get_ipython().magic(u'time unigram_all = easy_parallize(char_ldamodel,[revs_gen_all,revs_ungen_all])')
pickle.dump(bigram_all,open('tms_all_unigram_mp.p','wb'))

mod_gent = bigram_all[0] # bigram topic model for gentrified-area reviews 2010-2015
mod_ungent = bigram_all[1] # bigram topic model for gentrified-area reviews 2010-2015
g_all = output_topic_model(mod_gent,'tm_all_mp_bigram_10w_gent.csv',num_words)
ung_all = output_topic_model(mod_ungent,'tm_all_mp_bigram_10w_ungent.csv',num_words)


mod_gent = unigram_all[0] # unigram topic model for gentrified-area reviews 2010-2015
mod_ungent = unigram_all[1] # unigram topic model for gentrified-area reviews 2010-2015
g_all_u = output_topic_model(mod_gent,'tm_all_mp_unigram_10w_gent.csv',num_words)
ung_all_u = output_topic_model(mod_ungent,'tm_all_mp_unigram_10w_ungent.csv',num_words)


# In[12]:

print use_bigrams


# In[13]:

bigram_all = easy_parallize(char_ldamodel,[revs_gen_all,revs_ungen_all])


# In[16]:

pickle.dump(bigram_all,open('tms_all_bigrams_mp.p','wb'))


# In[18]:

num_words = 10
mod_gent = bigram_all[0] # bigram topic model for gentrified-area reviews 2010-2015
mod_ungent = bigram_all[1] # bigram topic model for gentrified-area reviews 2010-2015
g_all = output_topic_model(mod_gent,'tm_all_mp_bigram_10w_gent.csv',num_words)
ung_all = output_topic_model(mod_ungent,'tm_all_mp_bigram_10w_ungent.csv',num_words)


# In[7]:

allmodel = create_corpus(revs_gen_all + revs_ungen_all)


# In[24]:

num_words = 30
allmodel_readable = output_topic_model(mod_gent,'tm_all_GenAndUngen_bigram_10w_gent.csv',num_words)


# In[25]:

allmodel_readable


# In[ ]:



