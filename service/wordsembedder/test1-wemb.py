"""
Created on Wed Jan  2 22:28:48 2019

@author: scordoba
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import math
import nltk
import gensim
import logging

#from nltk.corpus import brown, movie_reviews, treebank


def get_corpus_from_string_list():
    txt = [
            ['Esto', 'es', 'una', 'frase', 'de', 'prueba.']
            , ['Otra', 'frase', 'de', 'prueba.']
            ]
    
    return txt



def get_corpus_from_file(file_path, type_encoding, char_phrase_split = '\n', char_token_split = ' '):
    f = open (file_path, encoding = type_encoding)
    
    txt = f.read().split('\n')
    txt = [nltk.word_tokenize(phrase) for phrase in txt]

    return txt



def compute_magnitude_of_word(model_wv, word):
    
    rep_vect = model_wv.wv[word]
    
    compute_sqr = lambda x: x ** 2
    rep_vect = compute_sqr(rep_vect)
    
    print( math.sqrt(sum(rep_vect)) )
    
    #return sum(test)
    

if __name__ == '__main__':
    
    # ...we load a text from file...
    newcorpus = get_corpus_from_file('corpus/text1.txt', 'utf8', '\n', ' ')
    #newcorpus = get_corpus_from_string_list()

    model = gensim.models.Word2Vec(newcorpus, min_count=2)
    #print(model.wv.vocab)
    #model.build_vocab(newcorpus) 
    
    words = list(model.wv.vocab)
    print(words)
    #print(model.wv.vocab)
    
    print(model.wv['espacio'])
    compute_magnitude_of_word(model, 'espacio')


