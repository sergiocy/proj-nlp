# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import logging
import os
import time
import numpy as np
import pandas as pd

#from app.lib.py.logging.create_logger import create_logger

# +
####
#### ...execution files...
PATH_LOG_FILE = 'log/log.log'

####
#### INPUT FILES
PATH_W2V_WORDS = '../00data/nlp/tmp/ws353_w2v_words.csv.gz'
#PATH_CHECKPOINT_W2V_WORDS_CONTEXT = '../00data/nlp/tmp/ws353_w2v_words_context.csv.gz'
PATH_W2V_DEF_DIRECT_SUM = '../00data/nlp/tmp/ws353_w2v_composed_direct_words_def_dict_sum.csv.gz'



#PATH_CHECKPOINT_BERT_WORDS = '../00data/nlp/tmp/ws353_bert_words.csv.gz'
#PATH_CHECKPOINT_BERT_WORDS_CONTEXT = '../00data/nlp/tmp/ws353_bert_words_context.csv.gz'


####
#### GLOBAL VARIABLES
w2v_vector_dimension = 300
w2v_vector_colnames = ['dim_{0}'.format(i) for i in range(1, w2v_vector_dimension + 1)]
bert_vector_dimension = 768
bert_vector_colnames = ['dim_{0}'.format(i) for i in range(1, bert_vector_dimension + 1)]
# -


start = time.time()
#os.remove(PATH_LOG_FILE)
#logger = create_logger(PATH_LOG_FILE)
#logger.info(' - starting execution')


# +
def compute_similarity_cosin(vector1, vector2):
    u = vector1 #array(vector1) 
    v = vector2 #array(vector2) 
    d = dot(u, v)
    c = dot(u,v)/norm(u)/norm(v) # -> cosine of the angle 
    angle = arccos(clip(c, -1, 1)) # if you really want the angle 

    return d, c, angle


def compute_similarities_vector_vs_set_vectors(in_vector
                                               , in_df_set_vectors):
    
# -



# ## W2V similarities



# +
w2v_words = pd.read_csv(PATH_W2V_WORDS, sep='|', header=0, compression='gzip')
w2v_words = w2v_words.drop(['id_token', 'token'], axis = 1)

#### ...we get a few words to dev...
w2v_words = w2v_words[w2v_words.id.isin([1, 2, 3])]

w2v_words.head()
# -

w2v_def_direct_sum = pd.read_csv(PATH_W2V_DEF_DIRECT_SUM, sep='|', header=0, compression='gzip')
w2v_def_direct_sum.head()

for index, row in w2v_words.iterrows():








