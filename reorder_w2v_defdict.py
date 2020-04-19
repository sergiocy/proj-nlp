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
                               , df_input = None
                               , col_words_sentence = None
                               , col_partition = None
                               , col_words_sentence_reordered = None
                               , type_order = 'syntactic'
                               , use_stanford_parser = True
                               , verbose = True):


    try:

        lst_ordered_words = list()

        for part in df_input[col_partition].unique():
            df_part = df_input[df_input[col_partition]==part]

            sentence = df_part[col_words_sentence].array
            sentence = list(sentence)

            if logger is not None:
                logger.info('reordering words in csv phrases')
                logger.info('input phrase: {}'.format(sentence))


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

            else:
                logger.info('this type_order not exist')
                raise Exception

            #### ...we build the output as list of phrases with reordered words...
            lst_ordered_words = lst_ordered_words + sentence


        #### ...we add to input dataset as new column...
        print('adding new columns')
        df_input[col_words_sentence_reordered] = np.asarray(lst_ordered_words)


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
                           , col_words_sentence = 'token'
                           , col_partition = 'w'
                           , col_words_sentence_reordered = 'token_reordered'#['id_token_reordered', 'token_reordered']
                           , type_order = 'reverse'
                           , use_stanford_parser = True
                           , verbose = True)


w2v_def_reordered
# -





# +
list1 = ["c", "b", "d", "a"]
list2 = [2, 3, 1, 4]

zipped_lists = zip(list1, list2)
print(zipped_lists)
#print(list(zipped_lists))

sorted_pairs = sorted(zipped_lists)
print(list(sorted_pairs))
tuples = zip(*sorted_pairs)
print(tuples)
list1, list2 = [list(tuple) for tuple in  tuples]
print(list1)
print(list2)


print ('#######')
print(zip(*zipped_lists))



# +
X = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
Y = [ 0,   1,   1,    0,   1,   2,   2,   0,   1]

paired_lists = zip(Y,X)
#print(paired_lists(0))

Z = [x for _,x in sorted(paired_lists)]
print(Z)  # ["a", "d", "h", "b", "c", "e", "i", "f", "g"]


#Y = [ 1,   1, 1,   2,   2, 1, 0, 0, 0]

Z = [x for _,x in zip(*paired_lists)]
print(Z)  

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

print(lst_reordered_token)
paired_reordered_token
# -






