
#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import math
import nltk
import gensim
#import logging
#from nltk.corpus import brown, movie_reviews, treebank




def compute_magnitude_of_word(model_wv, word):
    
    rep_vect = model_wv.wv[word]
    
    compute_sqr = lambda x: x ** 2
    rep_vect = compute_sqr(rep_vect)
    
    print( math.sqrt(sum(rep_vect)) )
    
    #return sum(test)
    

if __name__ == '__main__':
    ####
    #### INPUT PARAMS
    #### input text as list of words
    txt = ['esto', 'es', 'una', 'frase', 'de', 'prueba']
    
    model = gensim.models.Word2Vec(txt, min_count=2)
    print(model.wv.vocab)
    model.build_vocab(txt) 
    
    words = list(model.wv.vocab)
    print(words)
    #print(model.wv.vocab)
    
    print(model.wv['frase'])
    compute_magnitude_of_word(model, 'frase')


