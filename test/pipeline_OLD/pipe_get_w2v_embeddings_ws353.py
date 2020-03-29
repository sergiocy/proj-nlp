

def compute_embeddings():
    #################################################
    #### ...read file...
    data_def = load_input_text_csv(logger = logger
                            , new_colnames = ['w', 'def_dict', 'context']
                            , file_input = PATH_INPUT_DATA_DEF
                            , has_header = True
                            , sep = ';'
                            , encoding = 'utf-8'
                            , has_complete_rows = True
                            , cols_to_clean = ['w', 'def_dict', 'context']
                            , language = 'en'
                            , lcase = True
                            , lst_punct_to_del = ['\.', ',', '\(', '\)', ':', ';', '\?', '!', '"', '`']
                            , tokenized_text = False
                            , logging_tokens_cleaning = False
                            , insert_id_column = True
                            , inserted_id_column_name = 'id'
                            , file_save_pickle = None)

    logger.info(' - pandas dataframe cleaned; first rows...')
    logger.info('\n{0}'.format(data_def.loc[0:4]))


    rep_w2v = get_embedding_as_df(logger = None
                            , verbose = False
                            , df_input = data_def
                            , column_to_computing = 'w'
                            , columns_to_save = ['id', 'w']
                            , root_name_vect_cols = 'dim_'
                            , dim_embeddings = 300
                            , path_embeddings_model = PATH_W2V_MODEL
                            , type_model = 'W2V'
                            , python_pkg = 'gensim'
                            , file_save_pickle = PATH_CHECKPOINT_W2V_WORDS)




    rep_w2v = get_embedding_as_df(logger = None
                            , verbose = False
                            , df_input = data_def
                            , column_to_computing = 'context'
                            , columns_to_save = ['id', 'w']
                            , root_name_vect_cols = 'dim_'
                            , dim_embeddings = 300
                            , path_embeddings_model = PATH_W2V_MODEL
                            , type_model = 'W2V'
                            , python_pkg = 'gensim'
                            , file_save_pickle = PATH_CHECKPOINT_W2V_WORDS_CONTEXT)

    rep_w2v = get_embedding_as_df(logger = None
                            , verbose = False
                            , df_input = data_def
                            , column_to_computing = 'def_dict'
                            , columns_to_save = ['id', 'w']
                            , root_name_vect_cols = 'dim_'
                            , dim_embeddings = 300
                            , path_embeddings_model = PATH_W2V_MODEL
                            , type_model = 'W2V'
                            , python_pkg = 'gensim'
                            , file_save_pickle = PATH_CHECKPOINT_W2V_WORDS_DEF_DICT)
