# -*- coding: utf-8 -*-

import numpy as np
from numpy import (array, dot, arccos, clip) 
from numpy.linalg import norm 
from nltk.corpus import wordnet as wn
from scipy.stats.stats import pearsonr


####
#### similarity metric based on cosin between vector (for text vectorial representations)    
def compute_similarity_cosin(vector1, vector2):
    u = vector1 #array(vector1) 
    v = vector2 #array(vector2) 
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




    
if __name__ == '__main__':
    lst1 = ['a', 'girl', 'is', 'brushing', 'her', 'hair']
    lst2 = ['a', 'girl', 'is', 'styling', 'her', 'hair']
    print(lst1)
    print(lst2)

    
    
    
    
    
    
    
    
