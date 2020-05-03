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
from numpy import array, dot, arccos, clip 
from numpy.linalg import norm 
import pandas as pd


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
# +
#start = time.time()
#os.remove(PATH_LOG_FILE)
#logger = create_logger(PATH_LOG_FILE)
#logger.info(' - starting execution')
# -
# ## W2V VS BERT - ranking - DIRECT order - composition using POOLED

df_sim = pd.read_csv(PATH_SIM_BERT_DEF_REVERSE_POOLED, sep='|', header=0, compression='gzip')
#### ...we get a few words to dev...
#df_sim = df_sim[df_sim.id.isin([1, 2, 3])]
df_sim.head()



df_ranking_scores.to_csv(PATH_RANKING_BERT_DEF_REVERSE_POOLED_SCORES, sep='|', header=True, index=False, compression='gzip')
df_ranking_words.to_csv(PATH_RANKING_BERT_DEF_REVERSE_POOLED_WORDS, sep='|', header=True, index=False, compression='gzip')






