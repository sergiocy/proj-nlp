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

from app.lib.py.logging.create_logger import create_logger
from app.controller.operator.reorder_sentence_words_csv import reorder_sentence_words_csv

# +
####
#### ...execution files...
PATH_LOG_FILE = 'log/log.log'

####
#### data files during execution / checkpoints
PATH_CHECKPOINT_W2V_WORDS_DEFINITION = '../00data/nlp/tmp/ws353_w2v_words_def_dict.csv.gz'
PATH_CHECKPOINT_BERT_WORDS_DEFINITION = '../00data/nlp/tmp/ws353_bert_words_def_dict.csv.gz'

#######################################3
#### DATASETS PROCESSED - CHECKPOINTS PATHS...
PATH_CHECKPOINT_W2V_WORDS_DEFINITION_REORDERED_DIRECT = '../00data/nlp/tmp/ws353_w2v_reordered_direct_words_def_dict.csv.gz'
PATH_CHECKPOINT_W2V_WORDS_DEFINITION_REORDERED_REVERSE = '../00data/nlp/tmp/ws353_w2v_reordered_reverse_words_def_dict.csv.gz'
PATH_CHECKPOINT_BERT_WORDS_DEFINITION_REORDERED_DIRECT = '../00data/nlp/tmp/ws353_bert_reordered_direct_words_def_dict.csv.gz'
PATH_CHECKPOINT_BERT_WORDS_DEFINITION_REORDERED_REVERSE = '../00data/nlp/tmp/ws353_bert_reordered_reverse_words_def_dict.csv.gz'
# -
# ## LOAD W2V AND BUILD DATAFRAMES WITH REORDERED WORDS

# ### w2v dictionary definitions

# +
#start = time.time()
#os.remove(PATH_LOG_FILE)
#logger = create_logger(PATH_LOG_FILE)
#logger.info(' - starting execution')


#### DATASETS
#### ...pairs of words with mannual similarities...
w2v_def = pd.read_csv(PATH_CHECKPOINT_W2V_WORDS_DEFINITION, sep='|', header=0, compression='gzip')
print(len(w2v_def['id'].unique()))
print(len(w2v_def['w'].unique()))


w2v_vector_dimension = 300
#### ...get colnames with vector elements...
w2v_vector_colnames = ['dim_{0}'.format(i) for i in range(1, w2v_vector_dimension + 1)]


#### ...we get a few words to dev...
#w2v_def = w2v_def[w2v_def.id.isin([428])]

w2v_def.head(20)
# -



# +
w2v_def_reordered_direct = reorder_sentence_words_csv(logger = None #logger
                           , df_input = w2v_def
                           , cols_input_to_save = ['id', 'w']
                           , cols_vector = w2v_vector_colnames
                           , col_words_sentence = 'token'
                           , col_id_words_sentence = 'id_token'
                           , col_partition = 'w'
                           , col_words_sentence_reordered = 'token_reordered'#['id_token_reordered', 'token_reordered']
                           , col_id_words_sentence_reordered = 'id_token_reordered'
                           , type_order = 'direct'
                           , use_stanford_parser = True
                           , file_save_gz = PATH_CHECKPOINT_W2V_WORDS_DEFINITION_REORDERED_DIRECT 
                           , sep_out = '|' 
                           , verbose = True)


w2v_def_reordered_direct.head()
# +
w2v_def_reordered_reverse = reorder_sentence_words_csv(logger = None #logger
                           , df_input = w2v_def
                           , cols_input_to_save = ['id', 'w']
                           , cols_vector = w2v_vector_colnames
                           , col_words_sentence = 'token'
                           , col_id_words_sentence = 'id_token'
                           , col_partition = 'w'
                           , col_words_sentence_reordered = 'token_reordered'#['id_token_reordered', 'token_reordered']
                           , col_id_words_sentence_reordered = 'id_token_reordered'
                           , type_order = 'reverse'
                           , use_stanford_parser = True
                           , file_save_gz = PATH_CHECKPOINT_W2V_WORDS_DEFINITION_REORDERED_REVERSE 
                           , sep_out = '|' 
                           , verbose = True)


w2v_def_reordered_reverse
# -



# ## LOAD BERT AND BUILD DATAFRAMES WITH REORDERED WORDS

# +
#start = time.time()
#os.remove(PATH_LOG_FILE)
#logger = create_logger(PATH_LOG_FILE)
#logger.info(' - starting execution')


#### DATASETS
#### ...pairs of words with mannual similarities...
bert_def = pd.read_csv(PATH_CHECKPOINT_BERT_WORDS_DEFINITION, sep='|', header=0, compression='gzip')
print(len(bert_def['id'].unique()))
print(len(bert_def['w'].unique()))


bert_vector_dimension = 768
#### ...get colnames with vector elements...
bert_vector_colnames = ['dim_{0}'.format(i) for i in range(1, bert_vector_dimension + 1)]


#### ...we get a few words to dev...
#bert_def = bert_def[bert_def.id.isin([1, 2])]

bert_def.head(20)


# +
bert_def_reordered_direct = reorder_sentence_words_csv(logger = None #logger
                           , df_input = bert_def
                           , cols_input_to_save = ['id', 'w']
                           , cols_vector = bert_vector_colnames
                           , col_words_sentence = 'token'
                           , col_id_words_sentence = 'id_token'
                           , col_partition = 'w'
                           , col_words_sentence_reordered = 'token_reordered'#['id_token_reordered', 'token_reordered']
                           , col_id_words_sentence_reordered = 'id_token_reordered'
                           , type_order = 'direct'
                           , use_stanford_parser = True
                           , file_save_gz = PATH_CHECKPOINT_BERT_WORDS_DEFINITION_REORDERED_DIRECT 
                           , sep_out = '|' 
                           , verbose = True)


bert_def_reordered_direct.head()
# +
bert_def_reordered_direct = reorder_sentence_words_csv(logger = None #logger
                           , df_input = bert_def
                           , cols_input_to_save = ['id', 'w']
                           , cols_vector = bert_vector_colnames
                           , col_words_sentence = 'token'
                           , col_id_words_sentence = 'id_token'
                           , col_partition = 'w'
                           , col_words_sentence_reordered = 'token_reordered'#['id_token_reordered', 'token_reordered']
                           , col_id_words_sentence_reordered = 'id_token_reordered'
                           , type_order = 'reverse'
                           , use_stanford_parser = True
                           , file_save_gz = PATH_CHECKPOINT_BERT_WORDS_DEFINITION_REORDERED_REVERSE
                           , sep_out = '|' 
                           , verbose = True)


bert_def_reordered_direct.head()
# -






