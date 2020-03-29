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
PATH_INPUT_DATA_MEN_DEF_WN = '../00data/nlp/input/wordsim353/corpus_men_definitions_csv.csv'
PATH_INPUT_DATA_MEN_DEF_CLEANED_WN = '../00data/nlp/tmp/ws353_input_men_words_defwn.csv.gz'

####
#### data files during execution / checkpoints
#PATH_CHECKPOINT_BERT_MEN_WORDS = '../00data/nlp/tmp/ws353_bert_men_words.csv.gz'
#PATH_CHECKPOINT_BERT_MEN_ WORDS_CONTEXT = '../00data/nlp/tmp/ws353_bert_men_words_context.csv.gz'
PATH_CHECKPOINT_BERT_MEN_WORDS_DEF_WN = '../00data/nlp/tmp/ws353_bert_men_words_def_wn.csv.gz'
PATH_CHECKPOINT_W2V_MEN_WORDS_DEF_WN = '../00data/nlp/tmp/ws353_w2v_men_words_def_wn.csv.gz'
#PATH_CHECKPOINT_BERT_WORDS_DEF_WN = 'data/exchange/ws353_bert_def_wn'

# -




# +
start = time.time()

os.remove(PATH_LOG_FILE)
logger = create_logger(PATH_LOG_FILE)
logger.info(' - starting execution')
# -



# ## loading data


#### COMPUTING BERT-VECTORS OF WORDS-DEFINITION FROM WORDNET
data = load_input_text_csv(logger = logger
                        , new_colnames = ['id', 'w', 'def_wn', 'syntactic']
                        , file_input = PATH_INPUT_DATA_MEN_DEF_WN
                        , has_header = True
                        , sep_in = ';'
                        , encoding = 'utf-8'
                        , has_complete_rows = True
                        , cols_to_clean = ['w', 'def_wn']
                        , language = 'en'
                        , lcase = True
                        , lst_punct_to_del = ['\.', ',', '\(', '\)', ':', ';', '\?', '!', '"', '`']
                        , tokenized_text = False
                        , logging_tokens_cleaning = False
                        , insert_id_column = False
                        , inserted_id_column_name = 'id'
                        , file_save_pickle = None
                        , file_save_gz = PATH_INPUT_DATA_MEN_DEF_CLEANED_WN
                        , sep_out = '|')

#print(data.shape)
data.head()

#### ...computing bert representations for single words in dataset...
rep_w2v = get_embedding_as_df(logger = logger
                        , verbose = False
                        , df_input = data
                        , column_to_computing = 'def_wn'
                        , columns_to_save = ['id', 'w']
                        , root_name_vect_cols = 'dim_'
                        , dim_embeddings = 300
                        , path_embeddings_model = PATH_W2V_MODEL
                        , type_model = 'W2V'
                        , python_pkg = 'gensim'
                        , file_save_pickle = None
                        , file_save_gz = PATH_CHECKPOINT_W2V_MEN_WORDS_DEF_WN
                        , sep_out = '|')

rep_w2v.head()

#### ...computing bert representations for single words in dataset...
rep_bert = get_embedding_as_df(logger = logger
                        , verbose = False
                        , df_input = data
                        , column_to_computing = 'def_wn'
                        , columns_to_save = ['id', 'w']
                        , root_name_vect_cols = 'dim_'
                        , dim_embeddings = 768
                        , path_embeddings_model = None
                        , type_model = 'BERT'
                        , python_pkg = 'bert-embeddings'
                        , file_save_pickle = None
                        , file_save_gz = PATH_CHECKPOINT_BERT_MEN_WORDS_DEF_WN
                        , sep_out = '|')

rep_bert.head()









if logger is not None:
    logger.info('Process finished after {}'.format(time.time() - start))
else:
    print('Process finished after {}'.format(time.time() - start))
