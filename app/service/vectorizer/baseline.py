
#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import logging

import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
#import nltk
#import gensim
from gensim.models import Word2Vec

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



def plot_2d_vector(vec):
    #### to open plot; plt.show() works
    matplotlib.rcParams['interactive'] == True

    plt.plot(baseline_rep[0], baseline_rep[1])
    plt.show()
    #plt.savefig('figure.png')




if __name__ == '__main__':
    ####
    #### INPUT PARAMS
    #### input text as list of words
    #txt1 = ['esto', 'es', 'una', 'frase', 'de', 'prueba']
    #txt2 = ['this', 'is', 'a', 'test', 'phrase']

    sentences1 = [['esto', 'es', 'una', 'frase', 'de', 'prueba']
                    , ['esto', 'es', 'otra', 'frase']
                    , ['estamos', 'trabajando', 'en', 'representaciones', 'vectoriales']]
    #sentences2 = ['esto es una frase', 'esto es otra frase', 'estamos trabajando en representaciones vectoriales']
    #sentences3 = ['esto', 'es', 'una', 'frase', 'esto', 'es', 'otra', 'frase', 'estamos', 'trabajando', 'en', 'representaciones', 'vectoriales']
    #sentences4 = [['esto es una frase'], ['esto es otra frase'], ['estamos trabajando en representaciones vectoriales']]

    dim_representation = 2
    model, words_vocab = generating_model(sentences1
                                            , path_save_model='C:/sc-sync/projects/proj-nlp/config/model/model.bin'
                                            , rep_dimension=dim_representation)
    #new_model = Word2Vec.load('model.bin')
    print(model.wv['frase'])

    compute_magnitude_of_word(model, 'frase')

    print('#############')
    new_phrase = ['una', 'frase', 'de', 'prueba']
    print(new_phrase)
    baseline_rep = compute_average_vector_of_phrase(model, new_phrase, dim_representation)
    print('#############')
    print(baseline_rep)

    plot_2d_vector(baseline_rep)
    

