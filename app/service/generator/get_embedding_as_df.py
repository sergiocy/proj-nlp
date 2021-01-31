# -*- coding: utf-8 -*-

import traceback
import pandas as pd
import gensim

from app.service.vectorization.get_w2v_embedding_of_several_words_as_pd_df import *


####
#### FUNCTION TO GET A DATASET (df_input) AND PROCESS TEXT IN ONE OF ITS COLUMNS (with texts; a csv field with textual content)
#### (column_to_computing) GETTING ITS EMBEDDINGS AS PANDAS DF
def get_embedding_as_df(logger=None
                        , verbose=False
                        , df_input=None
                        , column_to_computing=None

                        , columns_to_save=[]
                        , root_name_vect_cols='dim_'
                        , dim_embeddings=300
                        , path_embeddings_model=None
                        , type_model='BERT'
                        # , python_pkg = None
                        , file_save_pickle=None
                        , file_save_gz=None
                        , sep_out='~'):
    #try:
    if column_to_computing is not None and df_input is not None:

        lst_embeddings = []

        if type_model == 'W2V':
            #### ...only gensim package...
            python_pkg = 'gensim'

            if python_pkg == 'gensim':
                model = gensim.models.KeyedVectors.load_word2vec_format(path_embeddings_model, binary=True)
                for it in df_input.index:
                    df_embeddings = get_w2v_embedding_of_several_words_as_pd_df(logger=logger
                                                                                , phrase_in=
                                                                                df_input[column_to_computing][it]
                                                                                , root_colnames=root_name_vect_cols
                                                                                , dim_vector_rep=dim_embeddings
                                                                                , embeddings_model=model)

                    #### ...insert columns to save in output...
                    #### we iter on inversed list to conserve the order of fields introduced
                    for c in reversed(columns_to_save):
                        df_embeddings.insert(0, c, [df_input[c][it] for i in range(len(df_embeddings))])

                    if logger is not None and verbose:
                        logger.info(df_embeddings)

                    lst_embeddings.append(df_embeddings)

                    #### ...to clean the flow and memory treatment...
                    del df_embeddings

                df_rep = pd.concat(lst_embeddings)

                #for c in columns_to_save:
                #    df_rep[c] = df_rep[c].astype('str')


        #### ...saving results as file...
        if file_save_pickle is not None:
            df_rep.to_pickle(file_save_pickle)
            if logger is not None:
                logger.info(" - data saved in file pickle {0}".format(file_save_pickle))
        if file_save_gz is not None:
            df_rep.to_csv(file_save_gz, sep=sep_out, header=True, index=False, compression='gzip')
            if logger is not None:
                logger.info(" - data saved in gz file {0}".format(file_save_gz))


    else:
        pass
        #traceback()
        #raise Exception

    #except Exception:
    #    if logger is not None:
    #        logger.exception("ERROR computing embeddings")
    #    raise Exception

    return df_rep
