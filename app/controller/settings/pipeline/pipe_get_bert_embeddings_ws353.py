
def compute_embeddings_ws353():

    ##################################
    #### READING FILES
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
                            , logging_tokens_cleaning = False)
    #### ...we add id for each row...
    data_def.insert(0, 'id', range(1, len(data_def) + 1))

    logger.info(' - pandas dataframe cleaned; first rows...')
    logger.info('\n{0}'.format(data_def.loc[0:4]))
    ####
    #### CHECKPOINT!! ...SERIALIZE INPUT DATASET AFTER LOAD AND CLEAN...
    #data_def.to_pickle(PATH_CHECKPOINT_INPUT)


    #################################################
    #### COMPUTING BERT-VECTORS OF SINGLE WORDS

    #data_def = pd.read_pickle(PATH_CHECKPOINT_INPUT)
    #data_def = data_def.iloc[0:4]

    lst_embed_words = []
    for iter in data_def.index:
        #### ...get embeddings for each word in a phrase as dataframe
        df_embeddings_word = get_bert_embedding_of_several_words_as_pd_df(logger = logger
                                                                        , phrase_in = data_def['w'][iter]
                                                                        , root_colnames = 'dim_w_'
                                                                        , dim_vector_rep = 768)
        #### ...insert id and word...
        df_embeddings_word.insert(0, 'w', [data_def['w'][iter] for i in range(len(df_embeddings_word))])
        df_embeddings_word.insert(0, 'id', [data_def['id'][iter] for i in range(len(df_embeddings_word))])
        #print(df_embeddings_word)

        lst_embed_words.append(df_embeddings_word)

    rep_words = pd.concat(lst_embed_words)
    rep_words.to_pickle(PATH_CHECKPOINT_BERT_WORDS)

    print(rep_words)
    #######################################################################
    ########################################################################


    #################################################
    #### COMPUTING BERT-VECTORS OF WORDS IN CONTEXT (PHRASES WITH CONTENTED WORD)
    #data_def = data_def.iloc[0:4]

    lst_embed_context = []
    for iter in data_def.index:
        #### ...get embeddings for each word in a phrase as dataframe
        df_embeddings_context = get_bert_embedding_of_several_words_as_pd_df(logger = logger
                                                                            , phrase_in = data_def['context'][iter]
                                                                            , root_colnames = 'dim_context_'
                                                                            , dim_vector_rep = 768)
        #### ...insert id and word...
        df_embeddings_context.insert(0, 'w', [data_def['w'][iter] for i in range(len(df_embeddings_context))])
        df_embeddings_context.insert(0, 'id', [data_def['id'][iter] for i in range(len(df_embeddings_context))])

        print(df_embeddings_context.iloc[:, 0:4])

        lst_embed_context.append(df_embeddings_context)

    rep_context = pd.concat(lst_embed_context)
    rep_context.to_pickle(PATH_CHECKPOINT_BERT_WORDS_CONTEXT)

    #print(rep_context)
    #######################################################################
    ########################################################################


    #################################################
    #### COMPUTING BERT-VECTORS OF WORD DEFINITIONS
    #data_def = data_def.iloc[0:4]
    #print(data_def)
    rep_def_dict = get_embedding_as_df(logger = logger
                            , verbose = True
                            , df_input = data_def
                            , column_to_computing = 'def_dict'
                            , columns_to_save = ['id', 'w']
                            , root_name_vect_cols = 'dim_def'
                            , dim_embeddings = 768
                            , embeddings_model = None
                            , type_model = 'BERT'
                            , file_save_pickle = PATH_CHECKPOINT_BERT_WORDS_DEF_DICT)


    #######################################################################
    ########################################################################
