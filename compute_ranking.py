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
def get_ranking_words_and_scores(df_sim
                                , cols_to_save = ['id', 'w']):

    ####
    #### ...define dataframes to save results...
    df_ranking_words = pd.DataFrame(columns = cols_to_save + ['pos_{0}'.format(i - len(cols_to_save) + 1) for i in range(len(cols_to_save), len(df_sim.columns))])
    df_ranking_scores = pd.DataFrame(columns = cols_to_save + ['pos_{0}'.format(i - len(cols_to_save) + 1) for i in range(len(cols_to_save), len(df_sim.columns))])

    for n_row, row in df_sim.iterrows():
        ####
        #### ...get values in columns to save...
        lst_columns_output = list(df_sim.loc[n_row][cols_to_save])

        ####
        #### ...get scores and words...
        lst_scores_and_words = zip(list(df_sim.drop(cols_to_save, axis = 1).loc[n_row])
                                  , list(df_sim.drop(cols_to_save, axis = 1).columns))
        lst_scores_and_words = list(lst_scores_and_words)

        ####
        #### ...and sort to get the ranking...
        lst_ranking = sorted(lst_scores_and_words, key = lambda x: x[0], reverse = True) 

        ####
        #### ...build rows for outputs dataframes...
        lst_ranking_scores = lst_columns_output + [lst_ranking[i][0] for i in range(len(lst_ranking))]
        lst_ranking_words = lst_columns_output + [lst_ranking[i][1] for i in range(len(lst_ranking))]

        df_ranking_scores.loc[n_row] = lst_ranking_scores
        df_ranking_words.loc[n_row] = lst_ranking_words

    return df_ranking_scores, df_ranking_words






# ## W2V - ranking - DIRECT order - composition using SUM

df_sim = pd.read_csv(PATH_SIM_W2V_DEF_DIRECT_SUM, sep='|', header=0, compression='gzip')
#### ...we get a few words to dev...
#df_sim = df_sim[df_sim.id.isin([1, 2, 3])]
df_sim.head()

# +
df_ranking_scores, df_ranking_words = get_ranking_words_and_scores(df_sim, cols_to_save = ['id', 'w'])

#df_ranking_scores.head()
df_ranking_words.head()
# -

df_ranking_scores.to_csv(PATH_RANKING_W2V_DEF_DIRECT_SUM_SCORES, sep='|', header=True, index=False, compression='gzip')
df_ranking_words.to_csv(PATH_RANKING_W2V_DEF_DIRECT_SUM_WORDS, sep='|', header=True, index=False, compression='gzip')

# ## W2V - ranking - DIRECT order - composition using AVG

df_sim = pd.read_csv(PATH_SIM_W2V_DEF_DIRECT_AVG, sep='|', header=0, compression='gzip')
#### ...we get a few words to dev...
#df_sim = df_sim[df_sim.id.isin([1, 2, 3])]
df_sim.head()

# +
df_ranking_scores, df_ranking_words = get_ranking_words_and_scores(df_sim, cols_to_save = ['id', 'w'])

#df_ranking_scores.head()
df_ranking_words.head()
# -

df_ranking_scores.to_csv(PATH_RANKING_W2V_DEF_DIRECT_AVG_SCORES, sep='|', header=True, index=False, compression='gzip')
df_ranking_words.to_csv(PATH_RANKING_W2V_DEF_DIRECT_AVG_WORDS, sep='|', header=True, index=False, compression='gzip')

# ## W2V - ranking - DIRECT order - composition using AVG-sequence

df_sim = pd.read_csv(PATH_SIM_W2V_DEF_DIRECT_AVG_SEQ, sep='|', header=0, compression='gzip')
#### ...we get a few words to dev...
#df_sim = df_sim[df_sim.id.isin([1, 2, 3])]
df_sim.head()

# +
df_ranking_scores, df_ranking_words = get_ranking_words_and_scores(df_sim, cols_to_save = ['id', 'w'])

#df_ranking_scores.head()
df_ranking_words.head()
# -

df_ranking_scores.to_csv(PATH_RANKING_W2V_DEF_DIRECT_AVG_SEQ_SCORES, sep='|', header=True, index=False, compression='gzip')
df_ranking_words.to_csv(PATH_RANKING_W2V_DEF_DIRECT_AVG_SEQ_WORDS, sep='|', header=True, index=False, compression='gzip')

# ## W2V - ranking - REVERSE order - composition using AVG-sequence

df_sim = pd.read_csv(PATH_SIM_W2V_DEF_REVERSE_AVG_SEQ, sep='|', header=0, compression='gzip')
#### ...we get a few words to dev...
#df_sim = df_sim[df_sim.id.isin([1, 2, 3])]
df_sim.head()

# +
df_ranking_scores, df_ranking_words = get_ranking_words_and_scores(df_sim, cols_to_save = ['id', 'w'])

#df_ranking_scores.head()
df_ranking_words.head()
# -

df_ranking_scores.to_csv(PATH_RANKING_W2V_DEF_REVERSE_AVG_SEQ_SCORES, sep='|', header=True, index=False, compression='gzip')
df_ranking_words.to_csv(PATH_RANKING_W2V_DEF_REVERSE_AVG_SEQ_WORDS, sep='|', header=True, index=False, compression='gzip')





# ## BERT - ranking - DIRECT order - composition using SUM

df_sim = pd.read_csv(PATH_SIM_BERT_DEF_DIRECT_SUM, sep='|', header=0, compression='gzip')
#### ...we get a few words to dev...
#df_sim = df_sim[df_sim.id.isin([1, 2, 3])]
df_sim.head()

# +
df_ranking_scores, df_ranking_words = get_ranking_words_and_scores(df_sim, cols_to_save = ['id', 'w'])

#df_ranking_scores.head()
df_ranking_words.head()
# -

df_ranking_scores.to_csv(PATH_RANKING_BERT_DEF_DIRECT_SUM_SCORES, sep='|', header=True, index=False, compression='gzip')
df_ranking_words.to_csv(PATH_RANKING_BERT_DEF_DIRECT_SUM_WORDS, sep='|', header=True, index=False, compression='gzip')

# ## BERT - ranking - DIRECT order - composition using AVG

df_sim = pd.read_csv(PATH_SIM_BERT_DEF_DIRECT_AVG, sep='|', header=0, compression='gzip')
#### ...we get a few words to dev...
#df_sim = df_sim[df_sim.id.isin([1, 2, 3])]
df_sim.head()

# +
df_ranking_scores, df_ranking_words = get_ranking_words_and_scores(df_sim, cols_to_save = ['id', 'w'])

#df_ranking_scores.head()
df_ranking_words.head()
# -

df_ranking_scores.to_csv(PATH_RANKING_BERT_DEF_DIRECT_AVG_SCORES, sep='|', header=True, index=False, compression='gzip')
df_ranking_words.to_csv(PATH_RANKING_BERT_DEF_DIRECT_AVG_WORDS, sep='|', header=True, index=False, compression='gzip')

# ## BERT - ranking - DIRECT order - composition using AVG-sequence

df_sim = pd.read_csv(PATH_SIM_BERT_DEF_DIRECT_AVG_SEQ, sep='|', header=0, compression='gzip')
#### ...we get a few words to dev...
#df_sim = df_sim[df_sim.id.isin([1, 2, 3])]
df_sim.head()

# +
df_ranking_scores, df_ranking_words = get_ranking_words_and_scores(df_sim, cols_to_save = ['id', 'w'])

#df_ranking_scores.head()
df_ranking_words.head()
# -

df_ranking_scores.to_csv(PATH_RANKING_BERT_DEF_DIRECT_AVG_SEQ_SCORES, sep='|', header=True, index=False, compression='gzip')
df_ranking_words.to_csv(PATH_RANKING_BERT_DEF_DIRECT_AVG_SEQ_WORDS, sep='|', header=True, index=False, compression='gzip')

# ## BERT - ranking - REVERSE order - composition using AVG-sequence

df_sim = pd.read_csv(PATH_SIM_BERT_DEF_REVERSE_AVG_SEQ, sep='|', header=0, compression='gzip')
#### ...we get a few words to dev...
#df_sim = df_sim[df_sim.id.isin([1, 2, 3])]
df_sim.head()

# +
df_ranking_scores, df_ranking_words = get_ranking_words_and_scores(df_sim, cols_to_save = ['id', 'w'])

#df_ranking_scores.head()
df_ranking_words.head()
# -

df_ranking_scores.to_csv(PATH_RANKING_BERT_DEF_REVERSE_AVG_SEQ_SCORES, sep='|', header=True, index=False, compression='gzip')
df_ranking_words.to_csv(PATH_RANKING_BERT_DEF_REVERSE_AVG_SEQ_WORDS, sep='|', header=True, index=False, compression='gzip')







# ## BERT - ranking - DIRECT order - composition using POOLED

df_sim = pd.read_csv(PATH_SIM_BERT_DEF_DIRECT_POOLED, sep='|', header=0, compression='gzip')
#### ...we get a few words to dev...
#df_sim = df_sim[df_sim.id.isin([1, 2, 3])]
df_sim.head()

# +
df_ranking_scores, df_ranking_words = get_ranking_words_and_scores(df_sim, cols_to_save = ['id', 'w'])

#df_ranking_scores.head()
df_ranking_words.head()
# -

df_ranking_scores.to_csv(PATH_RANKING_BERT_DEF_DIRECT_POOLED_SCORES, sep='|', header=True, index=False, compression='gzip')
df_ranking_words.to_csv(PATH_RANKING_BERT_DEF_DIRECT_POOLED_WORDS, sep='|', header=True, index=False, compression='gzip')

# ## BERT - ranking - REVERSE order - composition using POOLED

df_sim = pd.read_csv(PATH_SIM_BERT_DEF_REVERSE_POOLED, sep='|', header=0, compression='gzip')
#### ...we get a few words to dev...
#df_sim = df_sim[df_sim.id.isin([1, 2, 3])]
df_sim.head()

# +
df_ranking_scores, df_ranking_words = get_ranking_words_and_scores(df_sim, cols_to_save = ['id', 'w'])

#df_ranking_scores.head()
df_ranking_words.head()
# -

df_ranking_scores.to_csv(PATH_RANKING_BERT_DEF_REVERSE_POOLED_SCORES, sep='|', header=True, index=False, compression='gzip')
df_ranking_words.to_csv(PATH_RANKING_BERT_DEF_REVERSE_POOLED_WORDS, sep='|', header=True, index=False, compression='gzip')






