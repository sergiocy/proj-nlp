# -*- coding: utf-8 -*-

import pandas as pd

from service.vectorization.get_bert_embedding_of_several_words_as_pd_df import *

####
#### FUNCTION TO GET A DATASET (df_input) AND PROCESS TEXT IN ONE OF ITS COLUMNS
#### (column_to_computing) GETTING ITS EMBEDDINGS AS PANDAS DF
def get_embedding_as_df(logger = None
                        , verbose = False
                        , df_input = None
                        , column_to_computing = None
                        , columns_to_save = []
                        , root_name_vect_cols = 'dim_'
                        , dim_embeddings = 768
                        , embeddings_model = None
                        , type_model = 'BERT'
                        , file_save_pickle = None):

    try:
        if column_to_computing is not None and df_input is not None:

            lst_embeddings = []

            #### ...iteration on each cell in the selected column to process...s
            for iter in df_input.index:

                #### ...get embeddings for each word in a phrase as dataframe
                if type_model == 'BERT'
                    df_embeddings = get_bert_embedding_of_several_words_as_pd_df(logger = logger
                                                                                , phrase_in = df_input[column_to_computing][iter]
                                                                                , root_colnames = root_name_vect_cols
                                                                                , dim_vector_rep = dim_embeddings)
                #### ...insert columns to save in output...
                for c in columns_to_save:
                    df_embeddings.insert(0, c, [df_input[c][iter] for i in range(len(df_embeddings))])

                if logger is not None and verbose = True:
                    logger.info(df_embeddings)

                lst_embeddings.append(df_embeddings)

                #### ...to clean the flow and memory treatment...
                del df_embeddings

            df_rep = pd.concat(lst_embeddings)


            #### ...saving results as file...
            if file_save_pickle is not None:
                df_rep.to_pickle(file_save_pickle)


        else:
            raise Exception

    except Exception:
        if logger is not None:
            logger.exception("ERROR computing embeddings")
        raise Exception

    return df_rep
