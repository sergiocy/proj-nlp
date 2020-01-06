




def compute_embeddings():

    #################################################
    #### COMPUTING BERT-VECTORS OF WORDS-DEFINITION FROM WORDNET
    data_def = load_input_text_csv(logger = logger
                            , new_colnames = ['id', 'w', 'def_wn', 'syntactic']
                            , file_input = PATH_INPUT_DATA_DEF_WN
                            , has_header = True
                            , sep = ';'
                            , encoding = 'utf-8'
                            , has_complete_rows = True
                            , cols_to_clean = ['w', 'def_wn']
                            , language = 'en'
                            , lcase = True
                            , lst_punct_to_del = ['\.', ',', '\(', '\)', ':', ';', '\?', '!', '"', '`']
                            , tokenized_text = False
                            , logging_tokens_cleaning = False
                            , insert_id_column = False
                            #, inserted_id_column_name = 'id0'
                            , file_save_pickle = PATH_CHECKPOINT_INPUT_WORDNET)
    #### ...we add id for each row...
    logger.info(' - pandas dataframe cleaned; first rows...')
    logger.info('\n{0}'.format(data_def.loc[0:4]))
    #data_def = pd.read_pickle(PATH_CHECKPOINT_INPUT_WORDNET)


    #data_def = data_def.iloc[0:4]
    #print(data_def)

    rep_def_wn = get_embedding_as_df(logger = logger
                            , verbose = True
                            , df_input = data_def
                            , column_to_computing = 'def_wn'
                            , columns_to_save = ['id', 'w']
                            , root_name_vect_cols = 'dim_def_wn_'
                            , dim_embeddings = 768
                            , embeddings_model = None
                            , type_model = 'BERT'
                            , file_save_pickle = PATH_CHECKPOINT_BERT_WORDS_DEF_WN)



    #rep_def_wn = pd.read_pickle(PATH_CHECKPOINT_BERT_DEF_WORDNET)
    #print(rep_def_wn.head(10))
    #print(rep_def_wn.shape)
    #########################################################
    #########################################################
