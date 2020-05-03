# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
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
from numpy import array, dot, arccos, clip 
from numpy.linalg import norm 
import pandas as pd
import matplotlib.pyplot as plt


#from app.lib.py.logging.create_logger import create_logger

# +
####
#### ...execution files...
PATH_LOG_FILE = 'log/log.log'

####
#### INPUT FILES
PATH_RANKING_W2V_DEF_DIRECT_SUM_SCORES = '../00data/nlp/tmp/ranking/ws353_ranking_w2v_def_direct_sum_scores.csv.gz'
PATH_RANKING_W2V_DEF_DIRECT_SUM_WORDS = '../00data/nlp/tmp/ranking/ws353_ranking_w2v_def_direct_sum_words.csv.gz'
PATH_RANKING_W2V_DEF_DIRECT_AVG_SCORES = '../00data/nlp/tmp/ranking/ws353_ranking_w2v_def_direct_avg_scores.csv.gz'
PATH_RANKING_W2V_DEF_DIRECT_AVG_WORDS = '../00data/nlp/tmp/ranking/ws353_ranking_w2v_def_direct_avg_words.csv.gz'
PATH_RANKING_W2V_DEF_DIRECT_AVG_SEQ_SCORES = '../00data/nlp/tmp/ranking/ws353_ranking_w2v_def_direct_avg_seq_scores.csv.gz'
PATH_RANKING_W2V_DEF_DIRECT_AVG_SEQ_WORDS = '../00data/nlp/tmp/ranking/ws353_ranking_w2v_def_direct_avg_seq_words.csv.gz'
PATH_RANKING_W2V_DEF_REVERSE_AVG_SEQ_SCORES = '../00data/nlp/tmp/ranking/ws353_ranking_w2v_def_reverse_avg_seq_scores.csv.gz'
PATH_RANKING_W2V_DEF_REVERSE_AVG_SEQ_WORDS = '../00data/nlp/tmp/ranking/ws353_ranking_w2v_def_reverse_avg_seq_words.csv.gz'

PATH_RANKING_BERT_DEF_DIRECT_SUM_SCORES = '../00data/nlp/tmp/ranking/ws353_ranking_bert_def_direct_sum_scores.csv.gz'
PATH_RANKING_BERT_DEF_DIRECT_SUM_WORDS = '../00data/nlp/tmp/ranking/ws353_ranking_bert_def_direct_sum_words.csv.gz'
PATH_RANKING_BERT_DEF_DIRECT_AVG_SCORES = '../00data/nlp/tmp/ranking/ws353_ranking_bert_def_direct_avg_scores.csv.gz'
PATH_RANKING_BERT_DEF_DIRECT_AVG_WORDS = '../00data/nlp/tmp/ranking/ws353_ranking_bert_def_direct_avg_words.csv.gz'
PATH_RANKING_BERT_DEF_DIRECT_AVG_SEQ_SCORES = '../00data/nlp/tmp/ranking/ws353_ranking_bert_def_direct_avg_seq_scores.csv.gz'
PATH_RANKING_BERT_DEF_DIRECT_AVG_SEQ_WORDS = '../00data/nlp/tmp/ranking/ws353_ranking_bert_def_direct_avg_seq_words.csv.gz'
PATH_RANKING_BERT_DEF_REVERSE_AVG_SEQ_SCORES = '../00data/nlp/tmp/ranking/ws353_ranking_bert_def_reverse_avg_seq_scores.csv.gz'
PATH_RANKING_BERT_DEF_REVERSE_AVG_SEQ_WORDS = '../00data/nlp/tmp/ranking/ws353_ranking_bert_def_reverse_avg_seq_words.csv.gz'
PATH_RANKING_BERT_DEF_DIRECT_POOLED_SCORES = '../00data/nlp/tmp/ranking/ws353_ranking_bert_def_direct_pooled_scores.csv.gz'
PATH_RANKING_BERT_DEF_DIRECT_POOLED_WORDS = '../00data/nlp/tmp/ranking/ws353_ranking_bert_def_direct_pooled_words.csv.gz'
PATH_RANKING_BERT_DEF_REVERSE_POOLED_SCORES = '../00data/nlp/tmp/ranking/ws353_ranking_bert_def_reverse_pooled_scores.csv.gz'
PATH_RANKING_BERT_DEF_REVERSE_POOLED_WORDS = '../00data/nlp/tmp/ranking/ws353_ranking_bert_def_reverse_pooled_words.csv.gz'


####
#### GLOBAL VARIABLES
w2v_vr_dim = 300
bert_vr_dim = 768

w2v_pos_colnames = ['pos_{0}'.format(i) for i in range(1, 428)] 
bert_pos_colnames = ['pos_{0}'.format(i) for i in range(1, 424)] 
# +
#start = time.time()
#os.remove(PATH_LOG_FILE)
#logger = create_logger(PATH_LOG_FILE)
#logger.info(' - starting execution')
# -


# ## W2V - better composition function

# +
####
#### ...
df_scores = pd.read_csv(PATH_RANKING_W2V_DEF_DIRECT_SUM_SCORES, sep='|', header=0, compression='gzip')
df_words = pd.read_csv(PATH_RANKING_W2V_DEF_DIRECT_SUM_WORDS, sep='|', header=0, compression='gzip')

#df_scores.head()
#df_words.head()

lst_count_w2v_direct_sum = list()
for c in w2v_pos_colnames:
    lst_count_w2v_direct_sum.append(len(df_words[df_words['w'] == df_words[c]]))
# -

df_scores.head()

# +
####
#### ...
df_scores = pd.read_csv(PATH_RANKING_W2V_DEF_DIRECT_AVG_SCORES, sep='|', header=0, compression='gzip')
df_words = pd.read_csv(PATH_RANKING_W2V_DEF_DIRECT_AVG_WORDS, sep='|', header=0, compression='gzip')

#df_scores.head()
#df_words.head()

lst_count_w2v_direct_avg = list()
for c in w2v_pos_colnames:
    lst_count_w2v_direct_avg.append(len(df_words[df_words['w'] == df_words[c]]))
# -

df_scores.head()

### ...plot...
plt.figure(figsize=(16, 10))
#plt.plot(range(1, len(w2v_pos_colnames)+1), lst_count_w2v_direct_sum, label = 'w2v composition sum')
#plt.plot(range(1, len(bert_pos_colnames)+1), lst_count_bert_direct_sum, label = 'BERT composition sum')
plt.plot(range(1, 50), lst_count_w2v_direct_sum[0:49], label = 'w2v composition sum')
plt.plot(range(1, 50), lst_count_w2v_direct_avg[0:49], label = 'w2v composition avg')
plt.legend()
plt.title('ranking using SUM as composition function')
plt.show()

# ## W2V VS BERT - ranking - DIRECT order - composition SUM

# +
####
#### ...
df_scores = pd.read_csv(PATH_RANKING_W2V_DEF_DIRECT_SUM_SCORES, sep='|', header=0, compression='gzip')
df_words = pd.read_csv(PATH_RANKING_W2V_DEF_DIRECT_SUM_WORDS, sep='|', header=0, compression='gzip')

#df_scores.head()
#df_words.head()

lst_count_w2v_direct_sum = list()
for c in w2v_pos_colnames:
    lst_count_w2v_direct_sum.append(len(df_words[df_words['w'] == df_words[c]]))
# -

df_scores.head()

# +
df_scores = pd.read_csv(PATH_RANKING_BERT_DEF_DIRECT_SUM_SCORES, sep='|', header=0, compression='gzip')
df_words = pd.read_csv(PATH_RANKING_BERT_DEF_DIRECT_SUM_WORDS, sep='|', header=0, compression='gzip')

#df_scores.head()
#df_words.head()

lst_count_bert_direct_sum = list()
for c in bert_pos_colnames:
    lst_count_bert_direct_sum.append(len(df_words[df_words['w'] == df_words[c]]))
# -

df_scores.head()



### ...plot...
plt.figure(figsize=(16, 10))
#plt.plot(range(1, len(w2v_pos_colnames)+1), lst_count_w2v_direct_sum, label = 'w2v composition sum')
#plt.plot(range(1, len(bert_pos_colnames)+1), lst_count_bert_direct_sum, label = 'BERT composition sum')
plt.plot(range(1, 50), lst_count_w2v_direct_sum[0:49], label = 'w2v composition sum')
plt.plot(range(1, 50), lst_count_bert_direct_sum[0:49], label = 'BERT composition sum')
plt.legend()
plt.title('ranking using SUM as composition function')
plt.show()









