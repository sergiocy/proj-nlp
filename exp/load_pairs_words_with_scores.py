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
from app.controller.api import run_pipeline
from app.controller.operator.reorder_sentence_words_csv import reorder_sentence_words_csv

# +
####
#### ...execution files...
PATH_LOG_FILE = 'log/log.log'

####
#### ...models...
PATH_W2V_MODEL = '../00model/w2v/GoogleNews-vectors-negative300.bin'

####
#### CSV FILES - DATA INPUT
PATH_INPUT_DATA = '../00data/input/wordsim353/combined.csv'
#PATH_INPUT_DATA_DEF = '../../00data/input/wordsim353/combined-definitions.csv'
PATH_INPUT_DATA_DEF = '../00data/input/wordsim353/combined-definitions-context.csv'
PATH_INPUT_DATA_DEF_WN = '../00data/input/wordsim353/corpus_men_definitions_csv.csv'
PATH_INPUT_SIM_SCORES = '../00data/input/wordsim353/combined.csv'

####
#### data files during execution / checkpoints
PATH_CHECKPOINT_INPUT_SIM_PKL = '../00data/nlp/tmp/ws353_input_sim'
PATH_CHECKPOINT_INPUT_SIM_GZ = '../00data/nlp/tmp/ws353_input_sim.csv.gz'
PATH_CHECKPOINT_W2V_WORDS = '../00data/nlp/tmp/ws353_w2v_words.csv.gz'



#######################################3
#### PICKLE FILES - DATASETS PROCESSED - CHECKPOINTS PATHS...
PATH_CHECKPOINT_INPUT = 'data/exchange/ws353_input'
PATH_CHECKPOINT_INPUT_WORDNET = 'data/exchange/ws353_input_men_wordnet'

PATH_CHECKPOINT_BERT_WORDS_CONTEXT = 'data/exchange/ws353_bert_words_context'
PATH_CHECKPOINT_BERT_WORDS_DEF_DICT = 'data/exchange/ws353_bert_def_cambridge'
PATH_CHECKPOINT_BERT_WORDS_DEF_WN = 'data/exchange/ws353_bert_def_wn'

PATH_CHECKPOINT_W2V_WORDS = 'data/exchange/ws353_w2v_words'
PATH_CHECKPOINT_W2V_WORDS_CONTEXT = 'data/exchange/ws353_w2v_words_context'
PATH_CHECKPOINT_W2V_WORDS_DEF_DICT = 'data/exchange/ws353_w2v_def_cambridge'


####
#### DATA OUTPUT
PATH_OUTPUT_BERT_DATA_DEF = 'data/output/combined-definitions'
PATH_OUTPUT_BERT_DATA_COMPLETE = 'data/output/combined-definitions-complete'
PATH_OUTPUT_BERT_WORD_VS_DEF_1 = 'data/output/word_vs_def_1'
PATH_OUTPUT_BERT_WORD_VS_DEF_2 = 'data/output/word_vs_def_2'



CONFIG_PIPE_FILE_TEST = 'config/pipeline/config_pipe_test.ini'
# -




# +
start = time.time()

os.remove(PATH_LOG_FILE)
logger = create_logger(PATH_LOG_FILE)
logger.info(' - starting execution')
# -



# ## loading data


data = load_input_text_csv(logger = logger
                            , new_colnames = ['w1', 'w2', 'score']
                            , file_input = PATH_INPUT_SIM_SCORES
                            , has_header = True
                            , sep_in = ','
                            , encoding = 'utf-8'
                            , has_complete_rows = True
                            , cols_to_clean = ['w1', 'w2']
                            , language = 'en'
                            , lcase = True
                            , lst_punct_to_del = ['\.', ',', '\(', '\)', ':', ';', '\?', '!', '"', '`']
                            , tokenized_text = False
                            , logging_tokens_cleaning = False
                            , insert_id_column = False
                            #, inserted_id_column_name = 'id0'
                            , file_save_pickle = None
                            , file_save_gz = PATH_CHECKPOINT_INPUT_SIM_GZ
                            , sep_out = '|')

print(data.shape)
data.head()









if logger is not None:
    logger.info('Process finished after {}'.format(time.time() - start))
else:
    print('Process finished after {}'.format(time.time() - start))
