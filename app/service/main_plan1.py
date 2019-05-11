
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import logging

import numpy as np
import math
import pandas as pd
#import nltk
#from gensim.models import Word2Vec
import gensim

#import tensorflow as tf
#import tensorflow_hub as hub

#from nltk.corpus import brown, movie_reviews, treebank

from bert_embedding import BertEmbedding

#from numpy.core import *



def compute_magnitude_of_word(model_wv, word):
    
    rep_vect = model_wv.wv[word]
    print(word)
    print(rep_vect)
    
    compute_sqr = lambda x: x ** 2
    rep_vect = compute_sqr(rep_vect)
    
    magnitude = math.sqrt(sum(rep_vect))
    print(magnitude)
    
    return magnitude



def compute_average_vector_of_phrase(model_wv, phrase, rep_dimension): 
    sum_vector = np.zeros(shape=rep_dimension)
    print(sum_vector)

    for w in phrase:
        #print(w)
        print(model_wv.wv[w])
        #print(type(model.wv[w]))
        sum_vector = sum_vector + model_wv.wv[w]
        #print(sum_vector)

    print('----------')
    print(sum_vector)
    avg_vector = sum_vector/len(phrase)
    print(avg_vector)

    return avg_vector




if __name__ == '__main__':
    PATH_W2V_MODEL = '../../config/model/GoogleNews-vectors-negative300.bin'
    PATH_INPUT_DATA = '../../config/input/wordsim353/combined.csv'
    PATH_INPUT_DATA_DEF = '../../config/input/wordsim353/combined-definitions.csv'

    ##################################################
    #### READING FILE INPUT
    #data = pd.read_csv(PATH_INPUT_DATA)
    #### ...we select a subset to develope...
    #data = data.loc[0:10]
    #print(data)

    data_def = pd.read_csv(PATH_INPUT_DATA_DEF, sep=';', encoding='utf-8')
    data_def = data_def.loc[0:10]
    print(data_def)

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

    ############################################
    #### BERT model
    print('##########################')
    sen1 = data_def['definition'][0]
    print(sen1)



    bert_embedding = BertEmbedding()
    #result = bert_embedding(['something', 'to', 'encode'])
    #result = bert_embedding(['something to encode'])
    result = bert_embedding([sen1])
    print(type(sen1))

    first = result[0]

    #print(first[1])
    print(first[1][1].shape)


    
    
    
    

    


    
 
