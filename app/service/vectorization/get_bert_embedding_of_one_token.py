# -*- coding: utf-8 -*-


# import mxnet as mx #### ...to use GPU-utilities in 'bert_emnedding' package
from bert_embedding import BertEmbedding




####
#### FUNCTION TO GET A BERT-EMBEDDING OF A TOKEN FROM  "bert_embedding" (bert-embedding) package
#### and return a list of numpy-arrays representing each word in input_token (a phrase as string)
def get_bert_embedding_of_one_token(str_token_in, logger=None):

    try:

        str_token = str_token_in

        bert_embedding = BertEmbedding()
        lst_token = [str_token]

        if logger is not None:
            logger.info('input token: \'{0}\''.format(str_token_in))


        bert_rep = bert_embedding(lst_token)
        #print('###############################')
        #print(bert_rep)
        #print(bert_rep[0][1])

        #### ...el token...
        bert_rep_token_in = bert_rep[0][0]
        #print('processed token by bert-embedding package: {0}'.format(bert_processed_token))
        if logger is not None:
            logger.info('BERT input token: \'{0}\''.format(bert_rep_token_in))


        #### ...vectores...
        bert_rep_vector = bert_rep[0][1]
        #print(bert_rep_vector)
        #print(len(bert_rep_vector))
        #print(type(bert_rep_vector))
        #print(type(bert_rep_vector[0]))

    except Exception:
        if logger is not None:
            logger.exception("ERROR getting BERT-embedding of token \'{0}\'".format(str_token_in))
        #raise(SIGKILL)
        raise Exception

    return bert_rep_vector







if __name__=='__main__':
    lst1 = ['if', 'you', 'havent', 'you', 'should']
    lst2 = ['you', 'should', 'but', 'you', 'dont', 'have', 'to']

    print(get_bert_embedding_of_one_token('inmate', logger=None))
