# -*- coding: utf-8 -*-

import logging
import os
import time
import numpy as np
#import math
import pandas as pd
#from nltk.tokenize import word_tokenize
#from gensim.models import Word2Vec
#import gensim
#import tensorflow as tf
#import tensorflow_hub as hub

# import mxnet as mx #### ...to use GPU-utilities in 'bert_embedding' package
#from bert_embedding import BertEmbedding

#### imports personal modules
from lib.py.logging.create_logger import create_logger
from lib.py.datastructure.np_array_as_row_of_pd_df import np_array_as_row_of_pd_df
from controller.process.load_input_csv import read_csv
from controller.process.clean_phrase import clean_phrase
from service.vectorization.bert_vectorizer import get_bert_embedding_of_one_token

#from service.text.reader.read_csv import read_csv_and_add_or_change_colnames
#from service.computing.vector_computing import compute_vector_average_or_sum
#from service.computing.vector_similarity_metric import compute_similarity_cosin
#from service.computing.vector_similarity_metric import compute_pearson_coef



####
#### CSV FILES - DATA INPUT
PATH_LOG_FILE = '../log/log.log'
PATH_W2V_MODEL = '../config/model/GoogleNews-vectors-negative300.bin'
PATH_INPUT_DATA = '../../0-data/input/wordsim353/combined.csv'
#PATH_INPUT_DATA_DEF = '../../0-data/input/wordsim353/combined-definitions.csv'
PATH_INPUT_DATA_DEF = '../../0-data/input/wordsim353/combined-definitions-context.csv'

####
#### PICKLE FILES - DATASETS PROCESSED - CHECKPOINTS PATHS...
PATH_CHECKPOINT_INPUT = '../data/exchange/ws353_input'
PATH_CHECKPOINT_BERT_WORDS = '../data/exchange/ws353_bert_words'
#PATH_CHECKPOINT_BERT_DEF_CAMBRIDGE = '../data/exchange/ws353_bert_def_cambridge'
#PATH_CHECKPOINT_BERT_DEF_WORDNET = '../data/exchange/ws353_bert_def_wordnet'

####
#### DATA OUTPUT
PATH_OUTPUT_BERT_DATA_DEF = '../data/output/combined-definitions'
PATH_OUTPUT_BERT_DATA_COMPLETE = '../data/output/combined-definitions-complete'
PATH_OUTPUT_BERT_WORD_VS_DEF_1 = '../data/output/word_vs_def_1'
PATH_OUTPUT_BERT_WORD_VS_DEF_2 = '../data/output/word_vs_def_2'




if __name__ == '__main__':
    start = time.time()

    os.remove(PATH_LOG_FILE)
    logger = create_logger(PATH_LOG_FILE)
    logger.info(' - starting execution')



    ##################################
    ####
    #### READING FILES
    data_def = read_csv( logger = logger
                        , new_colnames = ['w', 'def_dict', 'context']
                        , file_input=PATH_INPUT_DATA_DEF
                        )

    #### ...we insert id in dataframe...
    data_def.insert(0, 'id', range(1, len(data_def)+1))


    #### ...developing with a few lines...
    data_def = data_def.loc[0:9]
    #### ...applying lambda function in data frame for each phrase
    data_def['w'] = data_def['w'].apply(lambda phrase: clean_phrase(phrase
                                                                                , language = 'en'
                                                                                , lcase=True
                                                                                , lst_punct_to_del = ['\.', ',', '\(', '\)', ':', ';', '\?', '!', '"', '`']
                                                                                , tokenized=False
                                                                                , logging_tokens_cleaning = False
                                                                                , logger = logger))

    data_def['def_dict'] = data_def['def_dict'].apply(lambda phrase: clean_phrase(phrase
                                                                                , language = 'en'
                                                                                , lcase=True
                                                                                , lst_punct_to_del = ['\.', ',', '\(', '\)', ':', ';', '\?', '!', '"', '`']
                                                                                , tokenized=False
                                                                                , logging_tokens_cleaning = False
                                                                                , logger = logger))

    data_def['context'] = data_def['context'].apply(lambda phrase: clean_phrase(phrase
                                                                                , language = 'en'
                                                                                , lcase=True
                                                                                , lst_punct_to_del = ['\.', ',', '\(', '\)', ':', ';', '\?', '!', '"', '`']
                                                                                , tokenized=False
                                                                                , logging_tokens_cleaning = False
                                                                                , logger = logger))


    logger.info(' - pandas dataframe cleaned; first rows...')
    logger.info('\n{0}'.format(data_def.loc[0:4]))
    ####
    #### CHECKPOINT!! ...SERIALIZE INPUT DATASET AFTER LOAD AND CLEAN...
    data_def.to_pickle(PATH_CHECKPOINT_INPUT)




    #################################################
    #### COMPUTING BERT-VECTORS OF SINGLE WORDS
    logger.info(' - BERT vectorizing...')
    #### get embedding as list...
    data_def['w_vect'] = data_def['w'].apply(lambda phrase: get_bert_embedding_of_one_token(phrase, logger=logger))
    #### select the first element in the list (only element because is an only word)
    data_def['w_vect'] = data_def['w_vect'].apply(lambda vector: vector[0])

    #### DATASET PREPARATION OF SINGLE WORDS
    #### ...structuring bert-rep as table...
    rep_w = data_def[['id', 'w', 'w_vect']]
    ####
    #### CHECKPOINT!! ...AFTER VECTORIZATION OF WORDS...
    #rep_w.to_pickle(PATH_CHECKPOINT_BERT_WORDS)

    df_vec = pd.concat([np_array_as_row_of_pd_df(logger = None
                                                    , np_array = rep_w['w_vect'][i]
                                                    , pd_colnames_root = 'dim_') for i in range(len(rep_w.index))])
    df_vec.index = rep_w.index
    rep_w = pd.concat([rep_w[['id', 'w']], df_vec], axis = 1)
    ####
    #### CHECKPOINT!! ...AFTER VECTORIZATION OF WORDS...
    rep_w.to_pickle(PATH_CHECKPOINT_BERT_WORDS)
    #rep_w = pd.read_pickle(PATH_CHECKPOINT_BERT_WORDS)
    #print(rep_w.head(10))
    #print(rep_w.shape)
    #######################################################################
    ########################################################################

    '''
    data_def['def_vectorized'] = data_def['def_cleaned'].apply(lambda phrase: get_bert_embedding_of_one_token(phrase, logger=logger))

    #### serializing dataframe as a pickle object
    data_def.to_pickle(PATH_OUTPUT_BERT_DATA_DEF)
    '''


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
