
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
#import numpy as np
#import math
import pandas as pd
#from nltk.tokenize import word_tokenize
#from gensim.models import Word2Vec
import gensim
#import tensorflow as tf
#import tensorflow_hub as hub

# import mxnet as mx #### ...to use GPU-utilities in 'bert_emnedding' package
from bert_embedding import BertEmbedding

#### imports personal modules
from service.util.utils import create_logger
from service.text.input_text.input_processing import clean_text, get_set_of_tokens
from service.vectorization.bert_vectorizer import get_bert_embedding_of_one_token
from service.computing.vector_computing import compute_mean_vector



PATH_LOG_FILE = '../log/log.log'
PATH_W2V_MODEL = '../config/model/GoogleNews-vectors-negative300.bin'
PATH_INPUT_DATA = '../config/input/wordsim353/combined.csv'
PATH_INPUT_DATA_DEF = '../config/input/wordsim353/combined-definitions.csv'
PATH_OUTPUT_BERT_DATA_DEF = '../config/output/combined-definitions'





def read_inputs(new_colnames
                , logger = None
                , file_input = PATH_INPUT_DATA_DEF
                , sep = ';'
                , encoding = 'utf-8'):
    ##################################################
    #### READING FILE INPUT
    #data_sim = pd.read_csv(PATH_INPUT_DATA, sep=',', encoding='utf-8')
    #data_sim.columns = ['w1', 'w2', 'sim']
    ##### ...we select a subset to develope...
    #data_sim = data_sim.loc[0:10]
    #print(data_sim)

    data_def = pd.read_csv(file_input, sep=sep, encoding=encoding)
    data_def.columns = new_colnames
    #### ...we select a subset to develope...
    #data_def = data_def.loc[0:2]

    return data_def

def process_inputs(df):
    ##################################################
    #### PREPROCESSING DEFINITIONS
    df['def_tokenized'] = df['def']
    df = clean_text(df, 'def_tokenized', lst_punt_to_del=['\.', ':', ';', '\?', '!', '"', '\'', '`', '=', ',', '\(', '\)'])
    df['def_cleaned'] = df['def_tokenized'].apply(lambda phrase: ' '.join(phrase))
    print(df)
    #print(get_set_of_tokens(list(data_def['def'])))

    return df

def compute_bert(data_def):
    ############################################
    #### BERT model
    print('######### BERT ############')
    w1 = data_def['w1'][0]
    def1 = data_def['def_cleaned'][0]
    print(w1)
    print(def1)

    bert_embedding = BertEmbedding()
    print('######### WORD ############')
    data_def['bert_word'] = data_def['w1'].apply(lambda phrase: get_bert_embedding_of_one_token(phrase, bert_embedding)[0])
    #print(bert_w1)
    
    print('######### DEFINITION ############')
    data_def['bert_definition'] = data_def['def_cleaned'].apply(lambda phrase: get_bert_embedding_of_one_token(phrase, bert_embedding))
    #bert_def1 = get_bert_embedding_of_one_token(def1, bert_embedding)
    #print(len(bert_def1))

    print(data_def)
    print(len(data_def['bert_word'][0]))
    print(len(data_def['bert_definition'][0]))
    print(type(data_def['bert_word'][0]))
    print(type(data_def['bert_definition'][0]))

    return(data_def)




if __name__ == '__main__':
    os.remove(PATH_LOG_FILE)
    logger = create_logger(PATH_LOG_FILE)
    logger.info(' - starting execution')

    #### READING FILES
    logger.info(' - read file {0}'.format(PATH_INPUT_DATA_DEF))
    data_def = read_inputs( logger = logger
                            , file_input = PATH_INPUT_DATA_DEF
                            , new_colnames = ['w1', 'def'])
    logger.info(' - dataframe loaded; first rows...')
    logger.info('\n{0}'.format(data_def.loc[0:2]))

    #data_def = process_inputs(data_def)
    #### COMPUTING BERT-VECTORS
    #data_def = compute_bert(data_def)
    #data_def.to_pickle(PATH_OUTPUT_BERT_DATA_DEF)


    #data_def = pd.read_pickle(PATH_OUTPUT_BERT_DATA_DEF)
    #data_def = data_def.loc[0:2]

    #print(data_def)
    #print(data_def.shape)


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





    
    
    
    

    


    
 
