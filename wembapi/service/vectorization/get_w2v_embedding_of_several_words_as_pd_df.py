# -*- coding: utf-8 -*-

import traceback

import pandas as pd
import gensim
#from gensim.models import Word2Vec

from app.lib.py.datastructure.np_array_as_row_of_pd_df import *


def get_w2v_embedding_of_several_words_as_pd_df(logger = None
                                                , phrase_in = None
                                                , root_colnames = 'dim_'
                                                , dim_vector_rep = 300
                                                , embeddings_model = None):
    try:
        if embeddings_model is not None:

            lst_phrase = list(gensim.utils.tokenize(phrase_in))

            if logger is not None:
                logger.info(' - computing W2V representation for input token: \'{0}\''.format(phrase_in))


            lst_embed_dfs = []
            lst_words_in_vocabulary = []
            for w in lst_phrase:
                #### ...CHECK if word exists in vocabulary model...
                if w in embeddings_model.vocab:
                    lst_words_in_vocabulary.append(w)
                    lst_embed_dfs.append(np_array_as_row_of_pd_df(logger = None
                                            , np_array = embeddings_model[w] #embeddings_model.wv[w]
                                            , pd_colnames_root = root_colnames))
                else:
                    if logger is not None:
                        logger.warn("word out-of-vocabulary - {0} REMOVED".format(w))

            rep_vect = pd.concat(lst_embed_dfs)

            ####
            #### ...insert columns as keys...
            rep_vect.insert(0, 'token', lst_words_in_vocabulary)
            rep_vect.insert(0, 'id_token', [i for i in range(1, len(lst_embed_dfs)+1)])

            ####
            #### ...reset dataframe index...
            rep_vect = rep_vect.reset_index(drop=True)

        else:
            raise Exception

    except Exception:
        if logger is not None:
            logger.exception("ERROR getting BERT-embedding of token \'{0}\'".format(phrase_in))
        raise Exception

    return rep_vect
