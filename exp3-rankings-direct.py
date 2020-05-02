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
PATH_SIM_W2V_DEF_DIRECT_SUM = '../00data/nlp/tmp/similarities_matrix/ws353_sim_w2v_def_direct_sum.csv.gz'
PATH_SIM_W2V_DEF_DIRECT_AVG = '../00data/nlp/tmp/similarities_matrix/ws353_sim_w2v_def_direct_avg.csv.gz'
PATH_SIM_W2V_DEF_DIRECT_AVG_SEQ = '../00data/nlp/tmp/similarities_matrix/ws353_sim_w2v_def_direct_avg_seq.csv.gz'
PATH_SIM_W2V_DEF_REVERSE_AVG_SEQ = '../00data/nlp/tmp/similarities_matrix/ws353_sim_w2v_def_reverse_avg_seq.csv.gz'

PATH_SIM_BERT_DEF_DIRECT_SUM = '../00data/nlp/tmp/similarities_matrix/ws353_sim_bert_def_direct_sum.csv.gz'
PATH_SIM_BERT_DEF_DIRECT_AVG = '../00data/nlp/tmp/similarities_matrix/ws353_sim_bert_def_direct_avg.csv.gz'
PATH_SIM_BERT_DEF_DIRECT_AVG_SEQ = '../00data/nlp/tmp/similarities_matrix/ws353_sim_bert_def_direct_avg_seq.csv.gz'
PATH_SIM_BERT_DEF_REVERSE_AVG_SEQ = '../00data/nlp/tmp/similarities_matrix/ws353_sim_bert_def_reverse_avg_seq.csv.gz'
PATH_SIM_BERT_DEF_DIRECT_POOLED = '../00data/nlp/tmp/similarities_matrix/ws353_sim_bert_def_direct_pooled.csv.gz'
PATH_SIM_BERT_DEF_REVERSE_POOLED = '../00data/nlp/tmp/similarities_matrix/ws353_sim_bert_def_reverse_pooled.csv.gz'

####
#### OUTPUT FILES
PATH_RANKING_W2V_DEF_DIRECT_SUM = '../00data/nlp/tmp/similarities_matrix/ws353_ranking_w2v_def_direct_sum.csv.gz'


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




# ## W2V - ranking - DIRECT order - composition using SUM

df_sim = pd.read_csv(PATH_SIM_W2V_DEF_DIRECT_SUM, sep='|', header=0, compression='gzip')
#### ...we get a few words to dev...
df_sim = df_sim[df_sim.id.isin([1, 2, 3])]
df_sim.head()

# +
cols_to_save = ['id', 'w']
n_row = 1

lst_columns_output = cols_to_save
df_ranking_words = pd.DataFrame(columns = cols_to_save + [i - len(cols_to_save) + 1 for i in range(len(cols_to_save), len(df_sim.columns))])
print(df_ranking_words)
df_ranking_scores = pd.DataFrame(columns = cols_to_save + [i - len(cols_to_save) + 1 for i in range(len(cols_to_save), len(df_sim.columns))])
print(df_ranking_scores)

lst_scores_and_words = zip(list(df_sim.drop(cols_to_save, axis = 1).loc[n_row])
                          , list(df_sim.drop(cols_to_save, axis = 1).columns))
lst_scores_and_words

# -



# +
#sim_w2v_direct_sum.to_csv(PATH_SIM_W2V_DEF_DIRECT_SUM, sep='|', header=True, index=False, compression='gzip')
