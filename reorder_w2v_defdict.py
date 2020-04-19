# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import logging
import os
import time
import numpy as np
import pandas as pd

from app.lib.py.logging.create_logger import create_logger

# +
####
#### ...execution files...
PATH_LOG_FILE = 'log/log.log'

####
#### data files during execution / checkpoints
PATH_CHECKPOINT_W2V_WORDS_DEFINITION = '../00data/nlp/tmp/ws353_w2v_words_def_dict.csv.gz'

#######################################3
#### PICKLE FILES - DATASETS PROCESSED - CHECKPOINTS PATHS...

# -



#### READING dataframe with phrases in one of its columns nd reorder to apply vector composition in this order
def reorder_sentence_words_csv(logger = None
                               , df_input = None # input dataset with words and its definition as column way in dataframe
                               , cols_input_to_save = None # list with columns to save from input dataset
                               , cols_vector = None # colnames where is the vector representetations...
                               , col_words_sentence = None # column with sentence words
                               , col_id_words_sentence = None # column with ids of words sentence
                               , col_partition = None
                               , col_words_sentence_reordered = None # column with reordered words as new column in dataframe 
                               , col_id_words_sentence_reordered = None # column with reordered words ids as new column in dataframe 
                               , type_order = 'syntactic'
                               , use_stanford_parser = True
                               , verbose = True):


    try:

        #lst_ordered_words = list()

        for part in df_input[col_partition].unique():
            #################################
            #### ...select partition associated with one word...
            df_part = df_input[df_input[col_partition]==part]

            ####
            #### ...we get words sentence and words ids as lists... 
            sentence = df_part[col_words_sentence].array
            sentence = list(sentence)
            id_token = list(df_part[col_id_words_sentence].array)
            
            print(sentence)
            print(id_token)
            
            ####
            #### ...paired input lists with tokens and its ids to reorder both later...
            paired_token = list(zip(sentence, id_token))
            
            

            if logger is not None:
                logger.info('reordering words in csv phrases')
                logger.info('input phrase: {}'.format(sentence))

            
            
            #################################
            #### ...reorder sentene words loaded in "sentence" variable as list of tokens...
            #### ...we implement different ways to reorder: 
            #### direct (no-reorder), reverse (reading from right to left), following syntactic structure
            if type_order == 'syntactic':
                
                pass

                #### TODO: code to up standford server API
                #### ...up syntactical parsin standford API..
                #java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &

                #sentence = reorder_syntactic_tokenized_sentence(logger = logger
                #                                                , lst_sentence = sentence
                #                                                , use_stanford_parser = True
                #                                                , verbose = verbose)

                #sentence = reorder_syntactic_tokenized_sentence_regex(logger = logger
                #                                                        , lst_sentence = sentence
                #                                                        , use_stanford_parser = True
                #                                                        , verbose = False)

            elif type_order == 'direct':
                sentence = sentence

            elif type_order == 'reverse':
                sentence.reverse()
                print(sentence)

            else:
                logger.info('this type_order not exist')
                raise Exception

               
            
            
            #################################
            #### ...we build the output as list of phrases with reordered words...
            lst_reordered_token = sentence
            lst_reordered_id_token = list()

            for reordered_token in lst_reordered_token:
                for i_pair in range(len(paired_token)):
                    if reordered_token == paired_token[i_pair][0]:
                        lst_reordered_id_token = lst_reordered_id_token + [paired_token[i_pair][1]]
                        del(paired_token[i_pair])
                        break
            
            print(lst_reordered_token)
            print(lst_reordered_id_token)
        
        
        
            #################################
            #### ...we add to input dataset as new column...
            print('adding new columns')
            #df_input[col_words_sentence_reordered] = np.asarray(lst_reordered_token)
            #df_input[col_id_words_sentence_reordered] = np.asarray(lst_reordered_id_token)

            ####
            #### ...generate dataset as input dataframe with reordered words, ids and vectors... 
            df_reordered = pd.DataFrame(data = df_part[cols_input_to_save])
            df_reordered[col_id_words_sentence_reordered] = np.asarray(lst_reordered_id_token)
            df_reordered[col_words_sentence_reordered] = np.asarray(lst_reordered_token)

            #print(df_reordered)
            #print(df_part)

            #### ...add vectors...
            #df_reordered = df_reordered.merge(df_part[cols_vector], left_on=[col_id_words_sentence_reordered, col_words_sentence_reordered]
            #                           , right_on=[col_id_words_sentence, col_words_sentence])

            df_reordered = pd.merge(df_reordered#.reset_index(drop = True)
                                    , df_part#[[cols_vector]]#.reset_index(drop = True)
                                    , left_on=[col_id_words_sentence_reordered, col_words_sentence_reordered]
                                    , right_on=[col_id_words_sentence, col_words_sentence]
                                    , how = 'left'
                                    , suffixes = ('', '_y'))

            print(df_reordered)


    except Exception:
        if logger is not None:
            logger.exception("ERROR reading csv")
        raise Exception


    return df_input



# ## LOAD W2V REPRESENTATIONS (DEFINITIONS) AND GLOBAL VARIABLES

# +
#start = time.time()
#os.remove(PATH_LOG_FILE)
#logger = create_logger(PATH_LOG_FILE)
#logger.info(' - starting execution')


#### DATASETS
#### ...pairs of words with mannual similarities...
w2v_def = pd.read_csv(PATH_CHECKPOINT_W2V_WORDS_DEFINITION, sep='|', header=0, compression='gzip')
print(len(w2v_def['id'].unique()))
print(len(w2v_def['w'].unique()))


w2v_vector_dimension = 300
#### ...get colnames with vector elements...
w2v_vector_colnames = ['dim_{0}'.format(i) for i in range(1, w2v_vector_dimension + 1)]


#### ...we get a few words to dev...
w2v_def = w2v_def.loc[0:8]

w2v_def.head(12)
# -



# +
w2v_def_reordered = reorder_sentence_words_csv(logger = None #logger
                           , df_input = w2v_def
                           , cols_input_to_save = ['id', 'w']
                           , cols_vector = w2v_vector_colnames
                           , col_words_sentence = 'token'
                           , col_id_words_sentence = 'id_token'
                           , col_partition = 'w'
                           , col_words_sentence_reordered = 'token_reordered'#['id_token_reordered', 'token_reordered']
                           , col_id_words_sentence_reordered = 'id_token_reordered'
                           , type_order = 'reverse'
                           , use_stanford_parser = True
                           , verbose = True)


w2v_def_reordered
# -







 

# +
token = ['i', 'love', 'you', 'with', 'too', 'love']
id_token = [1, 2, 3, 4, 5, 6] #


paired_token = list(zip(token, id_token))
print(paired_token)
#print([e[0] for e in paired_token])

lst_reordered_token = ['i', 'love', 'too', 'love', 'with', 'you']
lst_reordered_id_token = list()


for reordered_token in lst_reordered_token:
    for i_pair in range(len(paired_token)):
        if reordered_token == paired_token[i_pair][0]:
            lst_reordered_id_token = lst_reordered_id_token + [paired_token[i_pair][1]]
            del(paired_token[i_pair])
            break
                       
paired_reordered_token = list(zip(lst_reordered_token, lst_reordered_id_token))

paired_reordered_token = list(zip(lst_reordered_token, lst_reordered_id_token))
print(lst_reordered_token)
print(lst_reordered_id_token)

#print(lst_reordered_token)
#paired_reordered_token
# -






