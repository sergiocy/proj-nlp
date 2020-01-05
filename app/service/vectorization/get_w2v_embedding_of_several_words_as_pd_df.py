# -*- coding: utf-8 -*-

import traceback

import pandas as pd
from bert_embedding import BertEmbedding
import gensim
#from gensim.models import Word2Vec

from lib.py.datastructure.np_array_as_row_of_pd_df import *


def get_w2v_embedding_of_several_words_as_pd_df(logger = None
                                                , phrase_in = None
                                                , root_colnames = 'dim_'
                                                , dim_vector_rep = 300
                                                , embeddings_model = None):
    try:
        if embeddings_model is not None:

            lst_phrase = [phrase_in]
            colnames = ['{0}{1}'.format(root_colnames, i) for i in range(1, dim_vector_rep + 1)]

            if logger is not None:
                logger.info(' - computing W2V representation for input token: \'{0}\''.format(phrase_in))

            lst_embed_dfs = [np_array_as_row_of_pd_df(logger = None
                                        , np_array = embeddings_model.wv[w]
                                        , pd_colnames_root = 'dim_')  for w in lst_phrase]

            rep_vect = pd.concat(lst_embed_dfs)

            print("----------------------------------------")
            print(lst_embed_dfs)

            #print(embeddings_model['love'])
            #rep_vect = embeddings_model.wv['love']
            #wemb_sex = model.wv['sex']
            #print(rep_vect)
            #print(type(rep_vect))
            #print(len(rep_vect))

        else:
            raise Exception


        #bert_embedding = BertEmbedding()
        #bert_rep = bert_embedding(lst_phrase)

        #lst_words = bert_rep[0][0]
        #lst_bert_rep = bert_rep[0][1]

        #w_context_vect = pd.DataFrame(data = {'id_token': [i for i in range(1, len(lst_words) + 1)]
        #                                        , 'token': lst_words
        #                                        , 'bert': lst_bert_rep
        #                                        })

        #df_vect = pd.concat([pd.DataFrame(data = w_context_vect['bert'][i].reshape(1, len(w_context_vect['bert'][i]))
        #                                    , columns = colnames) for i in range(len(w_context_vect.index))])
        #df_vect.index = w_context_vect.index

        #w_context_vect = pd.concat([w_context_vect[['id_token', 'token']], df_vect], axis = 1)

    except Exception:
        if logger is not None:
            logger.exception("ERROR getting BERT-embedding of token \'{0}\'".format(phrase_in))
        raise Exception

    return rep_vect
