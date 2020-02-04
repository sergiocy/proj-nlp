# -*- coding: utf-8 -*-

import logging
import os
import time
import numpy as np
import pandas as pd

#import tensorflow as tf
#import tensorflow_hub as hub

# import mxnet as mx #### ...to use GPU-utilities in 'bert_embedding' package
#from bert_embedding import BertEmbedding

#### imports personal modules
from app.lib.py.logging.create_logger import create_logger

from app.controller.reader.load_input_text_csv import load_input_text_csv
from app.controller.generator.get_embedding_as_df import get_embedding_as_df
from app.controller.api import run_pipeline
#from service.vectorization.get_bert_embedding_of_several_words_as_pd_df import *

#from service.text.reader.read_csv import read_csv_and_add_or_change_colnames
#from service.computing.vector_computing import compute_vector_average_or_sum
#from service.computing.vector_similarity_metric import compute_similarity_cosin
#from service.computing.vector_similarity_metric import compute_pearson_coef



####
#### CSV FILES - DATA INPUT
PATH_LOG_FILE = 'log/log.log'
PATH_W2V_MODEL = '../0-model/w2v/GoogleNews-vectors-negative300.bin'
PATH_INPUT_DATA = '../0-data/input/wordsim353/combined.csv'
#PATH_INPUT_DATA_DEF = '../../0-data/input/wordsim353/combined-definitions.csv'
PATH_INPUT_DATA_DEF = '../0-data/input/wordsim353/combined-definitions-context.csv'
PATH_INPUT_DATA_DEF_WN = '../0-data/input/wordsim353/corpus_men_definitions_csv.csv'
PATH_INPUT_SIM_SCORES = '../0-data/input/wordsim353/combined.csv'


####
#### PICKLE FILES - DATASETS PROCESSED - CHECKPOINTS PATHS...
PATH_CHECKPOINT_INPUT = 'data/exchange/ws353_input'
PATH_CHECKPOINT_INPUT_WORDNET = 'data/exchange/ws353_input_men_wordnet'
PATH_CHECKPOINT_INPUT_SIM = 'data/exchange/ws353_input_sim'

PATH_CHECKPOINT_BERT_WORDS = 'data/exchange/ws353_bert_words'
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




if __name__ == '__main__':
    start = time.time()

    os.remove(PATH_LOG_FILE)
    logger = create_logger(PATH_LOG_FILE)
    logger.info(' - starting execution')



    run_pipeline(logger = logger, config_pipe_file = CONFIG_PIPE_FILE_TEST)



    '''
    ##################################
    #### READ BERT VECTOR AND JOIN WITH SIMILARITIES DATASET
    data_def = pd.read_pickle(PATH_OUTPUT_BERT_DATA_DEF)

    #### ...to dev, i get i few rows...
    #data_def = data_def.loc[0:2]

    #### ...joining with pairs of words similarities...
    #### ...load manual similarities file...
    data_sim = read_csv_and_add_or_change_colnames( logger = logger
                                                    , file_input = PATH_INPUT_DATA
                                                    , new_colnames = ['w1', 'w2', 'sim']
                                                    , sep = ','
                                                    )

    #### ...to dev, i get i few rows...
    #data_sim = data_sim.loc[0:2]

    data = pd.merge(data_sim, data_def, left_on='w1', right_on='w1', how='left')
    data.columns = ['w1', 'w2', 'sim', 'def1', 'def1_clean', 'w1_vectorized', 'def1_vectorized']
    data = pd.merge(data, data_def, left_on='w2', right_on='w1', how='left')
    data = data.drop(['w1_y'], axis=1)
    data.columns = ['w1', 'w2', 'sim', 'def1', 'def1_clean', 'w1_vectorized', 'def1_vectorized', 'def2', 'def2_clean', 'w2_vectorized', 'def2_vectorized']


    #print(data)
    #print(data.shape)
    #print(data.columns)
    #print(data[['w1', 'w2', 'w1_vectorized', 'def2', 'def2_vectorized']])
    #print(data_def[data_def['w1']=='cat'])

    #### serializing dataframe as a pickle object
    data.to_pickle(PATH_OUTPUT_BERT_DATA_COMPLETE)
    '''


    '''
    ##################################################
    #### COMPUTING SIMILARITIES BETWEEN word1-word2 FROM BERT VECTORS
    data = pd.read_pickle(PATH_OUTPUT_BERT_DATA_COMPLETE)
    data = data.dropna()

    #### ...to dev...
    #data = data.loc[0:10]
    print(data)
    print(data.shape)
    print(data.columns)
    print('--------')
    data['sim_w_w'] = data[['w1_vectorized', 'w2_vectorized']].apply(lambda r: compute_similarity_cosin(r['w1_vectorized'][0], r['w2_vectorized'][0])[1], axis=1)

    print(data[['w1', 'w2', 'sim', 'sim_w_w']])
    #print(data.shape)
    #print(data.columns)

    #### ...pearson correlation of cimilarities...
    print(compute_pearson_coef(np.asarray(data['sim']), np.asarray(data['sim_w_w']))[1])


    ##################################################
    #### COMPUTING AVG AND SUM VECTORS OF DEFINITIONS
    data['def1_vector_sum'] = data['def1_vectorized'].apply(lambda lst_vectors: compute_vector_average_or_sum(logger=logger, lst_np_arrays=lst_vectors, avg=False))
    data['def1_vector_avg'] = data['def1_vectorized'].apply(lambda lst_vectors: compute_vector_average_or_sum(logger=logger, lst_np_arrays=lst_vectors, avg=True))
    data['def2_vector_sum'] = data['def2_vectorized'].apply(lambda lst_vectors: compute_vector_average_or_sum(logger=logger, lst_np_arrays=lst_vectors, avg=False))
    data['def2_vector_avg'] = data['def2_vectorized'].apply(lambda lst_vectors: compute_vector_average_or_sum(logger=logger, lst_np_arrays=lst_vectors, avg=True))
    #print(data['def2_clean'].iloc[237:298])
    #print(data[data['def2_vectorized'][0][0]==-3.28788131e-01])
    print(data.shape)
    print(data.columns)

    #### serializing dataframe as a pickle object
    data.to_pickle(PATH_OUTPUT_BERT_DATA_COMPLETE)
    '''


    '''
    ##################################################
    #### COMPUTING AVG AND SUM VECTORS similarities
    data = pd.read_pickle(PATH_OUTPUT_BERT_DATA_COMPLETE)
    print(data.head())
    print(data.shape)
    print(data.columns)

    #### ...word1...
    data = data[['w1', 'w1_vectorized', 'def1_clean', 'def1_vector_sum', 'def1_vector_avg']].loc[0:5]
    '''








    if logger is not None:
        logger.info('Process finished after {}'.format(time.time() - start))
    else:
        print('Process finished after {}'.format(time.time() - start))
