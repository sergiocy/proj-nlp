
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import logging

import numpy as np
import math
import pandas as pd
#import nltk
from gensim.models import Word2Vec
import gensim

#from nltk.corpus import brown, movie_reviews, treebank



####
#### function to train a word2vec model with 'gensim' package
def generating_w2v_model(sentences, path_save_model=None, rep_dimension=5):
    
    model_emb = Word2Vec(sentences, min_count=1, size=rep_dimension)
    print(model_emb)

    words = list(model_emb.wv.vocab)
    print(words)
    #print(model_emb.wv.vocab)
    #print(model_emb.wv[words[0]])

    if path_save_model is not None:
        model_emb.save(path_save_model)

    return model_emb, words


####
#### function to get a gensim (word2Vec) model builded and saved in path 'path_model_to_load'
def get_model(path_model_to_load):
    return Word2Vec.load(path_model_to_load)






if __name__ == '__main__':
    PATH_W2V_MODEL = '../../config/model/GoogleNews-vectors-negative300.bin'
    PATH_INPUT_DATA = '../../config/input/wordsim353/combined.csv'
    PATH_INPUT_DATA_DEF = '../../config/input/wordsim353/combined.csv'

    ###model = get_model(PATH_MODEL)
    ###model = gensim.models.Word2Vec.load_word2vec_format(PATH_MODEL, binary=True)  
    model = gensim.models.KeyedVectors.load_word2vec_format(PATH_W2V_MODEL, binary=True)


    data = pd.read_csv(PATH_INPUT_DATA)


    #print(data)
    word1 = data['Word 1']
    print(word1)

    # TEST WITH TWO WORDS
    #wemb_love = model.wv['love']
    #wemb_sex = model.wv['sex']
    #print(wemb_love)
    
    
    

    


    
 
