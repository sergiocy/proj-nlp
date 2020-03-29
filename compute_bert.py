# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
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

#import tensorflow as tf
#import tensorflow_hub as hub

from app.lib.py.logging.create_logger import create_logger

from app.controller.reader.load_input_text_csv import load_input_text_csv
from app.controller.generator.get_embedding_as_df import get_embedding_as_df
#from app.controller.api import run_pipeline
#from app.controller.operator.reorder_sentence_words_csv import reorder_sentence_words_csv

# +
####
#### ...execution files...
PATH_LOG_FILE = 'log/log.log'

####
#### ...models...
PATH_W2V_MODEL = '../00model/w2v/GoogleNews-vectors-negative300.bin'

####
#### CSV FILES - DATA INPUT
#PATH_INPUT_DATA_DEF = '../00data/nlp/input/wordsim353/combined-definitions-context.csv'
PATH_INPUT_DATA_DEF_CLEANED = '../00data/nlp/tmp/ws353_input_words_defdict_context.csv.gz'


####
#### data files during execution / checkpoints
PATH_CHECKPOINT_BERT_WORDS = '../00data/nlp/tmp/ws353_bert_words.csv.gz'
PATH_CHECKPOINT_BERT_WORDS_CONTEXT = '../00data/nlp/tmp/ws353_bert_words_context.csv.gz'
PATH_CHECKPOINT_BERT_WORDS_DEF_DICT = '../00data/nlp/tmp/ws353_bert_words_def_dict.csv.gz'
#PATH_CHECKPOINT_BERT_WORDS_DEF_WN = 'data/exchange/ws353_bert_def_wn'

# -




# +
start = time.time()

os.remove(PATH_LOG_FILE)
logger = create_logger(PATH_LOG_FILE)
logger.info(' - starting execution')
# -



# ## loading data


####
#### gz with input data generated in flow to compute w2v representations...
data = pd.read_csv(PATH_INPUT_DATA_DEF_CLEANED, sep='|', header=0, compression='gzip')

#print(data.shape)
data.head()

#### ...computing bert representations for single words in dataset...
rep_bert = get_embedding_as_df(logger = logger
                        , verbose = False
                        , df_input = data
                        , column_to_computing = 'w'
                        , columns_to_save = ['id', 'w']
                        , root_name_vect_cols = 'dim_'
                        , dim_embeddings = 768
                        , path_embeddings_model = None
                        , type_model = 'BERT'
                        , python_pkg = 'bert-embeddings'
                        , file_save_pickle = None
                        , file_save_gz = PATH_CHECKPOINT_BERT_WORDS
                        , sep_out = '|')

rep_bert.head()

#### ...computing bert representations for words in context...
rep_bert = get_embedding_as_df(logger = logger
                        , verbose = False
                        , df_input = data
                        , column_to_computing = 'context'
                        , columns_to_save = ['id', 'w']
                        , root_name_vect_cols = 'dim_'
                        , dim_embeddings = 768
                        , path_embeddings_model = None
                        , type_model = 'BERT'
                        , python_pkg = 'bert-embeddings'
                        , file_save_pickle = None
                        , file_save_gz = PATH_CHECKPOINT_BERT_WORDS_CONTEXT
                        , sep_out = '|')



#### ...computing bert representations for word definitions...
rep_bert = get_embedding_as_df(logger = logger
                        , verbose = False
                        , df_input = data
                        , column_to_computing = 'def_dict'
                        , columns_to_save = ['id', 'w']
                        , root_name_vect_cols = 'dim_'
                        , dim_embeddings = 768
                        , path_embeddings_model = None
                        , type_model = 'BERT'
                        , python_pkg = 'bert-embeddings'
                        , file_save_pickle = None
                        , file_save_gz = PATH_CHECKPOINT_BERT_WORDS_DEF_DICT
                        , sep_out = '|')



if logger is not None:
    logger.info('Process finished after {}'.format(time.time() - start))
else:
    print('Process finished after {}'.format(time.time() - start))
