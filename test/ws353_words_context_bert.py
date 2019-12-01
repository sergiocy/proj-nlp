# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd

#### ...for function to get BERT-embedding of several words as pandas dataframe...
from bert_embedding import BertEmbedding
# -



####
#### FUNCTION TO GET A BERT-EMBEDDING OF A TOKEN FROM  "bert_embedding" (bert-embedding) package
#### and return a list of numpy-arrays representing each word in input_token (a phrase as string) 
def get_bert_embedding_of_one_token(str_token_in, logger=None):

    try:

        str_token = str_token_in

        bert_embedding = BertEmbedding()
        lst_token = [str_token]

        if logger is not None:
            logger.info('input token: \'{0}\''.format(str_token_in))


        bert_rep = bert_embedding(lst_token)
        #print('###############################')
        #print(bert_rep)
        #print(bert_rep[0][1])

        #### ...el token...
        bert_rep_token_in = bert_rep[0][0]
        #print('processed token by bert-embedding package: {0}'.format(bert_processed_token))
        if logger is not None:
            logger.info('BERT input token: \'{0}\''.format(bert_rep_token_in))


        #### ...vectores...
        bert_rep_vector = bert_rep[0][1]
        #print(bert_rep_vector)
        #print(len(bert_rep_vector))
        #print(type(bert_rep_vector))
        #print(type(bert_rep_vector[0]))

    except Exception:
        if logger is not None:
            logger.exception("ERROR getting BERT-embedding of token \'{0}\'".format(str_token_in))
        raise Exception

    return bert_rep_vector





# +
PATH_CHECKPOINT_INPUT = '../../data/exchange/ws353_input'
data_def = pd.read_pickle(PATH_CHECKPOINT_INPUT)

data_def.head(10)
#data_def.shape

# +
##############################################################
################# DEVELOPING THE FUNCTION ####################
##############################################################

# +

from bert_embedding import BertEmbedding


def get_bert_embedding_of_several_words_as_pd_df(logger = None
                                                , phrase_in = None
                                                , root_colnames = 'dim_'
                                                , dim_vector_rep = 768):
    try:
        lst_phrase = [phrase_in]
        colnames = ['{0}{1}'.format(pd_colnames_root, i) for i in range(1, dim_vector_rep + 1)]
        
        if logger is not None:
            logger.info(' - computing BERT representation for input token: \'{0}\''.format(phrase_in))
        
        bert_embedding = BertEmbedding()
        bert_rep = bert_embedding(lst_phrase)
        
        lst_words = bert_rep[0][0]
        lst_bert_rep = bert_rep[0][1]
        
        w_context_vect = pd.DataFrame(data = {'id_context': [i for i in range(1, len(lst_words) + 1)]
                                                , 'w_context': lst_words
                                                , 'w_context_bert': lst_bert_rep
                                                })
        
        df_vect = pd.concat([pd.DataFrame(data = w_context_vect['w_context_bert'][i].reshape(1, len(w_context_vect['w_context_bert'][i]))
                                            , columns = colnames) for i in range(len(w_context_vect.index))])
        df_vect.index = w_context_vect.index

        w_context_vect = pd.concat([w_context_vect[['id_context', 'w_context']], df_vect], axis = 1)
        
    except Exception:
        if logger is not None:
            logger.exception("ERROR getting BERT-embedding of token \'{0}\'".format(phrase_in))
        raise Exception

    return w_context_vect


# -

get_bert_embedding_of_several_words_as_pd_df(logger = None
                                                , phrase_in = data_def['context'][1]
                                                , root_colnames = 'dim_context_'
                                                , dim_vector_rep = 768)





# +
str_token = data_def['context'][0] #str_token_in
print(str_token)

#### ...i need to use the prase as list... if not, by default the bert_embedding package will tokenize the phrase as LIST OF LETTERS (CHARACTERS)
lst_token = [str_token]
print(lst_token)


# +
bert_embedding = BertEmbedding()
bert_rep = bert_embedding(lst_token)

#### ...bert_embedding pkg tokenize the list element by default, returning one bert-vector by word/token...
#print(bert_rep)
print(type(bert_rep))
print(len(bert_rep))
#print(bert_rep)

print(type(bert_rep[0]))
print(len(bert_rep[0]))

# +
#for tok in range(len(bert_rep)):
lst_words = bert_rep[0][0]
lst_bert_rep = bert_rep[0][1]

#print(lst_words)
#print(lst_bert_rep)
#print(data)

w_context_vect = pd.DataFrame(data = {'id_context': [i for i in range(1, len(lst_words) + 1)]
        , 'w_context': lst_words
        , 'w_context_bert': lst_bert_rep
       })

print(w_context_vect)
print(type(w_context_vect['w_context_bert'][0]))
print(w_context_vect['w_context_bert'])

# +
pd_colnames_root = 'dim_context_'
dim_bert_rep = len(w_context_vect['w_context_bert'][0])

colnames = ['{0}{1}'.format(pd_colnames_root, i) for i in range(1, dim_bert_rep + 1)]
#print(colnames)

df_vect = pd.concat([pd.DataFrame(data = w_context_vect['w_context_bert'][i].reshape(1, len(w_context_vect['w_context_bert'][i]))
              , columns = colnames) for i in range(len(w_context_vect.index))])
df_vect.index = w_context_vect.index

w_context_vect = pd.concat([w_context_vect[['id_context', 'w_context']], df_vect], axis = 1)
print(w_context_vect)
