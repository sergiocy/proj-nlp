# -*- coding: utf-8 -*-

import pandas as pd
from bert_embedding import BertEmbedding


def get_bert_embedding_of_several_words_as_pd_df(logger = None
                                                , phrase_in = None
                                                , root_colnames = 'dim_'
                                                , dim_vector_rep = 768):
    try:
        lst_phrase = [phrase_in]
        colnames = ['{0}{1}'.format(root_colnames, i) for i in range(1, dim_vector_rep + 1)]

        if logger is not None:
            logger.info(' - computing BERT representation for input token: \'{0}\''.format(phrase_in))

        bert_embedding = BertEmbedding()
        bert_rep = bert_embedding(lst_phrase)

        lst_words = bert_rep[0][0]
        lst_bert_rep = bert_rep[0][1]

        w_context_vect = pd.DataFrame(data = {'id_token': [i for i in range(1, len(lst_words) + 1)]
                                                , 'token': lst_words
                                                , 'bert': lst_bert_rep
                                                })

        df_vect = pd.concat([pd.DataFrame(data = w_context_vect['bert'][i].reshape(1, len(w_context_vect['bert'][i]))
                                            , columns = colnames) for i in range(len(w_context_vect.index))])
        df_vect.index = w_context_vect.index

        w_context_vect = pd.concat([w_context_vect[['id_token', 'token']], df_vect], axis = 1)

    except Exception:
        if logger is not None:
            logger.exception("ERROR getting BERT-embedding of token \'{0}\'".format(phrase_in))
        raise Exception

    return w_context_vect
