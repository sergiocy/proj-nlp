# -*- coding: utf-8 -*-

import logging
import os
#import numpy as np
#import math
import pandas as pd
#from nltk.tokenize import word_tokenize
#from gensim.models import Word2Vec
#import gensim
#import tensorflow as tf
#import tensorflow_hub as hub

# import mxnet as mx #### ...to use GPU-utilities in 'bert_emnedding' package
#from bert_embedding import BertEmbedding

#### imports personal modules
from service.util.utils import create_logger
#from service.text.cleaner.clean_df_text import clean_text_column
from service.text.cleaner.clean_text import clean_phrase
#from service.text.reader.input_processing import get_set_of_tokens
from service.text.reader.read_csv import read_csv_and_add_or_change_colnames
from service.vectorization.bert_vectorizer import get_bert_embedding_of_one_token
from service.computing.vector_computing import compute_mean_vector



PATH_LOG_FILE = '../log/log.log'
PATH_W2V_MODEL = '../config/model/GoogleNews-vectors-negative300.bin'
PATH_INPUT_DATA = '../data/input/wordsim353/combined.csv'
PATH_INPUT_DATA_DEF = '../data/input/wordsim353/combined-definitions.csv'
PATH_OUTPUT_BERT_DATA_DEF = '../data/output/combined-definitions'








if __name__ == '__main__':
    os.remove(PATH_LOG_FILE)
    logger = create_logger(PATH_LOG_FILE)
    logger.info(' - starting execution')
    '''
    ##################################
    ####
    #### READING FILES
    data_def = read_csv_and_add_or_change_colnames( logger = logger
                                                    , file_input = PATH_INPUT_DATA_DEF
                                                    , new_colnames = ['w1', 'def']
                                                    )

    ####
    #### CLEANING PHRASES IN CSV
    data_def["def"].fillna("", inplace = True) 
    data_def = data_def[data_def['def'] != '']

    #### ...developing with a few lines...
    #data_def = data_def.loc[0:10]
    #### ...applying lambda function in data frame for each phrase
    data_def['def_cleaned'] = data_def['def'].apply(lambda phrase: clean_phrase(phrase
                                                                                , language = 'en'
                                                                                , lcase=True
                                                                                , lst_punct_to_del = ['\.', ',', '\(', '\)', ':', ';', '\?', '!', '"', '`']
                                                                                , tokenized=False
                                                                                , logging_tokens_cleaning = False
                                                                                , logger = logger))

    logger.info(' - pandas dataframe clean (tokenized or not); first rows...')
    logger.info('\n{0}'.format(data_def.loc[0:10]))

    ####
    #### COMPUTING BERT-VECTORS
    logger.info(' - BERT vectorizing...')
    data_def['w1_vectorized'] = data_def['w1'].apply(lambda phrase: get_bert_embedding_of_one_token(phrase, logger=logger)) 
    data_def['def_vectorized'] = data_def['def_cleaned'].apply(lambda phrase: get_bert_embedding_of_one_token(phrase, logger=logger)) 

    #### serializing dataframe as a pickle object
    data_def.to_pickle(PATH_OUTPUT_BERT_DATA_DEF)
    '''


    ##################################
    #### READ BERT VECTOR AND COMPUTING SIMILARITIES BETWEEN word-definition
    data_def = pd.read_pickle(PATH_OUTPUT_BERT_DATA_DEF)
    data_def = data_def.loc[0:2]

    print(data_def)
    print(data_def.shape)



    ##################################################
    #### ...load Google Word2Vec model...
    #model = gensim.models.KeyedVectors.load_word2vec_format(PATH_W2V_MODEL, binary=True)

    #print(data)
    #word1 = data['Word 1']
    #print(word1)

    # TEST WITH TWO WORDS
    #wemb_love = model.wv['love']
    #wemb_sex = model.wv['sex']
    #print(wemb_love)



    ##################################################
    #### ...load manual similarities file...
    #data_sim = read_csv_and_add_or_change_colnames( logger = logger
    #                                                , file_input = PATH_INPUT_DATA
    #                                                , new_colnames = ['w1', 'w2', 'sim']
    #                                                , sep = ','
    #                                                )





    
    
    
    

    


    
 
