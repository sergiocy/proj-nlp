# -*- coding: utf-8 -*-

import numpy as np
from numpy import (array, dot, arccos, clip) 
from numpy.linalg import norm 
from nltk.corpus import wordnet as wn
from scipy.stats.stats import pearsonr


####
#### similarity metric based on cosin between vector (for text vectorial representations)    
def compute_similarity_cosin(vector1, vector2):
    u = array(vector1) 
    v = array(vector2) 
    d = dot(u, v)
    c = dot(u,v)/norm(u)/norm(v) # -> cosine of the angle 
    angle = arccos(clip(c, -1, 1)) # if you really want the angle 

    return d, c, angle  


####
#### function to compute pearson coefficient between two variables
#### input: two numpy-arrays
def compute_pearson_coef(vector1, vector2):
    
    #### ...two ways to compute pearson coefficient...
    pearson1 = pearsonr(vector1, vector2)
    pearson2 = np.corrcoef(vector1, vector2)[0, 1]
    
    #print(pearson1)
    #print(pearson2)
    return pearson1, pearson2


####
#### FUNCTION TO GET SIMILARITY BETWEEN TWO WORDS USING WordNet SIMILARITY 
def get_wordnet_words_similarity(word1, word2, type_score = 'path_similarity', corpus_ic = None):    
    word_synsets_1 = wn.synsets(word1)
    word_synsets_2 = wn.synsets(word2)
    #print(word_synsets_1)
    #print(word_synsets_2)
    
    scores = []
    
    for syn1 in word_synsets_1:
        for syn2 in word_synsets_2:
            if type_score is 'path_similarity':
                scores.append(syn1.path_similarity(syn2))
            if type_score is 'res':
                scores.append(syn1.res_similarity(syn2, corpus_ic))
            if type_score is 'jcn':
                scores.append(syn1.jcn_similarity(syn2, corpus_ic))
            if type_score is 'lin':
                scores.append(syn1.lin_similarity(syn2, corpus_ic))
    #print(scores)           
    return scores


####
#### FUNCTION TO GET NUMBER OF WORDS OVERLAPPED (SAME WORDS) BETWEEN SENTENCES AS WORDS-LIST
def get_number_words_overlapped(lst_sen1, lst_sen2):

    count = [(1 if w in lst_sen2 else 0) for w in lst_sen1] 
    count=sum(count)
    
    return count


####
#### FUNCTION TO GET NUMBER OF WORDS OVERLAPPED (SAME WORDS) BETWEEN SENTENCES AS WORDS-LIST
def get_number_synsets_overlapped(lst_sen1, lst_sen2):
    
    lst_flag_overlapped = []
    for w1 in lst_sen1:
        lst_w1_synsets = wn.synsets(w1)
        #print(lst_w1_synsets)
        for w2 in lst_sen2:
            lst_w2_synsets = wn.synsets(w2)
            #print(lst_w2_synsets)
            
            count_syn = [(1 if w1_syn in lst_w2_synsets else 0) for w1_syn in lst_w1_synsets] 
            
            if sum(count_syn) >= 1:
                lst_flag_overlapped.append(1)
            else:
                lst_flag_overlapped.append(0)
        
    n_overlapped = sum(lst_flag_overlapped)
    
    return lst_flag_overlapped, n_overlapped



    
if __name__ == '__main__':
    lst1 = ['a', 'girl', 'is', 'brushing', 'her', 'hair']
    lst2 = ['a', 'girl', 'is', 'styling', 'her', 'hair']
    print(lst1)
    print(lst2)
    number = get_number_words_overlapped(lst1, lst2)
    print('number is {0}'.format(number))
    
    lst_flag_overlapped, n_overlapped = get_number_synsets_overlapped(lst1, lst2)
    print(lst_flag_overlapped)
    print(n_overlapped)
    
    
    ######
    vector1 = [0, 0, 0, 0, 0, 0, 0, 0]
    vector2 = [0, 0, 0, 0, 0, 0, 1.0, 1.0]
    print(compute_similarity_cosin(vector1, vector2))
    
    lst1 = ['people', 'are', 'dancing', 'outside']
    lst2 = ['a', 'group', 'of', 'people', 'are', 'dancing'] 
    get_wordnet_words_similarity('havent', 'but', type_score = 'res', corpus_ic = None)
    
    
    
    
    
    
    
    
    
