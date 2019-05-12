# -*- coding: utf-8 -*-


# import mxnet as mx #### ...to use GPU-utilities in 'bert_emnedding' package
from bert_embedding import BertEmbedding

  
        

####
#### FUNCTION TO GET A TOKEN AS STRING AND AN OBJECT "bert_embedding" from bert-embedding package
#### and return a list of vectors 
def get_bert_embedding_of_one_token(str_token, bert_embedding):

    #bert_embedding = BertEmbedding()

    lst_token = [str_token]
    print('input token: {0}'.format(lst_token))

    #### ....phrase to test...
    #lst_token = ['something to encode']

    bert_w1 = bert_embedding(lst_token)
    #print(bert_w1)

    #### ...el token...
    bert_processed_token = bert_w1[0][0]
    print('processed token by bert-embedding package: {0}'.format(bert_processed_token))

    #### ...vectores...
    vector_rep = bert_w1[0][1]
    #print(vector_rep)
    #print(len(vector_rep))
    #print(type(vector_rep))
    #print(type(vector_rep[0]))

    return vector_rep







if __name__=='__main__':
    lst1 = ['if', 'you', 'havent', 'you', 'should']
    lst2 = ['you', 'should', 'but', 'you', 'dont', 'have', 'to']  
    

    

