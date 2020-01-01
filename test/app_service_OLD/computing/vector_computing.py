# -*- coding: utf-8 -*-


# import mxnet as mx #### ...to use GPU-utilities in 'bert_emnedding' package
from bert_embedding import BertEmbedding
import numpy as np
import math
        


#### FUNCTION TO COMPUTE AVERAGE VECTOR FROM A LIST OF NUMPY-ARRAYS
####  ...these can represent the vectors associated to each word of a phrase or text-fragment
#### ...if avg=False, sum is computed, if is True, average is computed...
def compute_vector_average_or_sum(logger=None, lst_np_arrays=None, avg=False):
    
    try:
        #if logger is not None:
        #    logger.info('computing vector average')

        #### ...we need to have two vectors almost...
        if len(lst_np_arrays)>=2:
            array_cum = np.add(lst_np_arrays[0], lst_np_arrays[1])
        
            if len(lst_np_arrays)>2:
                for index_np_array in range(2, len(lst_np_arrays)):
                    array_cum = np.add(array_cum, lst_np_arrays[index_np_array])

            if avg==True:
                lambda_fun = lambda elem: elem/len(lst_np_arrays)
                array_cum = lambda_fun(array_cum)

        else:
            if len(lst_np_arrays)==1:
                array_cum = lst_np_arrays[0]
            else:
                raise Exception


        return array_cum


    except Exception:
        if logger is not None:
            logger.exception("ERROR computing vector average")
            logger.info("{0}".format(lst_np_arrays))
        raise Exception

    
    




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

    




if __name__=='__main__':
    lst1 = ['if', 'you', 'havent', 'you', 'should']
    lst2 = ['you', 'should', 'but', 'you', 'dont', 'have', 'to']  
    

    

