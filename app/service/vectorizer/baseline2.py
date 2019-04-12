
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import logging

import numpy as np
import math
import pandas as pd
#import nltk
#import gensim
from gensim.models import Word2Vec
import gensim

#from nltk.corpus import brown, movie_reviews, treebank



####
#### function to train a word2vec model
def generating_model(sentences, path_save_model=None, rep_dimension=5):
    
    model_emb = Word2Vec(sentences, min_count=1, size=rep_dimension)
    print(model_emb)

    words = list(model_emb.wv.vocab)
    print(words)
    #print(model_emb.wv.vocab)
    #print(model_emb.wv[words[0]])

    if path_save_model is not None:
        model_emb.save(path_save_model)

    return model_emb, words


def get_model(path_model_to_load):
    return Word2Vec.load(path_model_to_load)




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
        print(model.wv[w])
        #print(type(model.wv[w]))
        sum_vector = sum_vector + model.wv[w]
        #print(sum_vector)

    print('----------')
    print(sum_vector)
    avg_vector = sum_vector/len(phrase)
    print(avg_vector)

    return avg_vector




if __name__ == '__main__':
    PATH_MODEL = '../config/model/GoogleNews-vectors-negative300.bin'
    PATH_INPUT_DATA = '../config/input/wordsim353/combined.csv'


    ###model = get_model(PATH_MODEL)
    ###model = gensim.models.Word2Vec.load_word2vec_format(PATH_MODEL, binary=True)  
    model = gensim.models.KeyedVectors.load_word2vec_format(PATH_MODEL, binary=True)


    data = pd.read_csv(PATH_INPUT_DATA)
    #print(data)
    word1 = data['Word 1']
    #print(word1)

    # TEST WITH TWO WORDS
    wemb_love = model.wv['love']
    wemb_sex = model.wv['sex']
    print(wemb_love)


    

