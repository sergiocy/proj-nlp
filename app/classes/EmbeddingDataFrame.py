import logging
import os
import time
import numpy as np
import pandas as pd

from app.service.generator.get_embedding_as_df import get_embedding_as_df
from app.service.reader.load_input_text_csv import load_input_text_csv


class EmbeddingDataFrame:

    #### attributes


    #### constructor
    def __init__(self
                # parameters if build=False ; a matrix with embeddings is readen
                , filepath=None
                , logger=None
                , build=False
                 # parameters if build=True ; a matrix with embeddings is builded
                 # from csv with text pieces
                 , colname_text=None
                 , filepath_prepared_text=None
                 , filepath_embeddings=None
                 , type_model=None
                 , path_model=None
                 , dim_embedding=None
                 , root_name_vect_cols=None):

        #### ...we validate input arguments/attributes in constructor...
        #### ...logger and filepath required...
        if logger is None or filepath is None:
            raise
        else:
            self.logger = logger
            self.build = build
            self.filepath = filepath
            self.filepath_prepared_text = filepath_prepared_text
            self.filepath_embeddings = filepath_embeddings
            self.colname_text = colname_text
            self.type_model = type_model
            self.path_model = path_model
            self.root_name_vect_cols = root_name_vect_cols

            self.df_emb = None


        if build:
            if self.filepath_prepared_text is None \
                    or self.colname_text is None \
                    or self.type_model is None \
                    or self.path_model is None \
                    or self.root_name_vect_cols is None:
                raise
            else:
                self.filepath_prepared_text = filepath_prepared_text
                self.colname_text = colname_text
                self.type_model = type_model
                self.path_model = path_model
                self.filepath_embeddings = filepath_embeddings
                self.dim_embedding = dim_embedding
        else:
            #### we receive a file with embeddings
            if self.filepath_embeddings is None \
                    or self.self.dim_embedding is None\
                    or self.root_name_vect_cols is None:
                raise
            else:
                self.filepath_embeddings = filepath_embeddings
                self.dim_embedding = dim_embedding
                self.root_name_vect_cols = root_name_vect_cols

    def get_embeddings(self):
        if self.build:
            #### we receive a file as csv file with strings/sentences

            #### ...load and clena texts...
            self.df_emb = load_input_text_csv(logger=self.logger
                                     , file_input=self.filepath
                                     , cols_to_clean=self.colname_text
                                     , file_save_gz=self.filepath_prepared_text)

            #### ...prepare embeddings DataFrame...
            self.df_emb = get_embedding_as_df(logger=self.logger
                                     , verbose=False
                                     , df_input=self.df_emb
                                     , column_to_computing=self.colname_text[0]
                                     , columns_to_save=self.df_emb.columns
                                     , root_name_vect_cols=self.root_name_vect_cols
                                     , dim_embeddings=self.dim_embedding
                                     , path_embeddings_model=self.path_model
                                     , type_model=self.type_model
                                     , file_save_gz=self.filepath_embeddings)
        else:
            self.logger.info('leemos fichero con embeddings')
