

def compute_embeddings():
    
    #################################################
    #### COMPUTING BERT-VECTORS OF WORDS-DEFINITION FROM WORDNET
    data_def = load_input_text_csv(logger = logger
                            , new_colnames = ['w1', 'w2', 'score']
                            , file_input = PATH_INPUT_SIM_SCORES
                            , has_header = True
                            , sep = ','
                            , encoding = 'utf-8'
                            , has_complete_rows = True
                            , cols_to_clean = ['w1', 'w2']
                            , language = 'en'
                            , lcase = True
                            , lst_punct_to_del = ['\.', ',', '\(', '\)', ':', ';', '\?', '!', '"', '`']
                            , tokenized_text = False
                            , logging_tokens_cleaning = False
                            , insert_id_column = False
                            #, inserted_id_column_name = 'id0'
                            , file_save_pickle = PATH_CHECKPOINT_INPUT_SIM)
    #### ...we add id for each row...
    logger.info(' - pandas dataframe cleaned; first rows...')
    logger.info('\n{0}'.format(data_def.loc[0:4]))
    #data_def = pd.read_pickle(PATH_CHECKPOINT_INPUT_WORDNET)
