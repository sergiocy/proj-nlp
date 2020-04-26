# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import Model 
#import bert
from bert.tokenization import bert_tokenization
# +
# See BERT paper: https://arxiv.org/pdf/1810.04805.pdf
# And BERT implementation convert_single_example() at https://github.com/google-research/bert/blob/master/run_classifier.py

def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids


def get_bert_tensorflow_hub_model(max_seq_length = 128
                                 , module_hub_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"):
    
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids") 

    bert_layer = hub.KerasLayer(module_hub_url, trainable=True)
    #bert_layer = hub.KerasLayer("C:/sc/sync/projects/00model/bert/uncased_new", trainable=True)

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])
    tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
    
    return model, tokenizer
# -



# ## GLOBAL VARIABLES

# +
####
#### ...execution files...
PATH_LOG_FILE = 'log/log.log'

####
#### INPUT DATA FILES (ouput of reordering process)
#PATH_CHECKPOINT_W2V_WORDS_DEFINITION_REORDERED_DIRECT = '../00data/nlp/tmp/ws353_w2v_reordered_direct_words_def_dict.csv.gz'
#PATH_CHECKPOINT_W2V_WORDS_DEFINITION_REORDERED_REVERSE = '../00data/nlp/tmp/ws353_w2v_reordered_reverse_words_def_dict.csv.gz'
PATH_CHECKPOINT_BERT_WORDS_DEFINITION_REORDERED_DIRECT = '../00data/nlp/tmp/ws353_bert_reordered_direct_words_def_dict.csv.gz'
PATH_CHECKPOINT_BERT_WORDS_DEFINITION_REORDERED_REVERSE = '../00data/nlp/tmp/ws353_bert_reordered_reverse_words_def_dict.csv.gz'

####
#### OUTPUT
PATH_CHECKPOINT_BERT_WORDS_DEFINITION_POOLED_DIRECT = '../00data/nlp/tmp/ws353_bert_pooled_direct_words_def_dict.csv.gz'
PATH_CHECKPOINT_BERT_WORDS_DEFINITION_POOLED_REVERSE = '../00data/nlp/tmp/ws353_bert_pooled_reverse_words_def_dict.csv.gz'
# -

# ## LOAD BERT DATA - order direct
#
# This procedure is only available for BERT model. 
#
# We will apply it in sentences ordering in direct and reverse way.



# +
#start = time.time()
#os.remove(PATH_LOG_FILE)
#logger = create_logger(PATH_LOG_FILE)
#logger.info(' - starting execution')


#### LOAD DATASET
#### ...pairs of words with mannual similarities...
bert_def = pd.read_csv(PATH_CHECKPOINT_BERT_WORDS_DEFINITION_REORDERED_DIRECT, sep='|', header=0, compression='gzip')

####
#### DEFINE VECTOR REPRESENTATION COLNAMES
bert_vector_dimension = 768
#### ...get colnames with vector elements...
bert_vector_colnames = ['dim_{0}'.format(i) for i in range(1, bert_vector_dimension + 1)]


#### ...we get a few words to dev...
#bert_def = bert_def[bert_def.id.isin([428])]

bert_def.head(12)
# -





# ## BUILD AND USE MODEL - direct order

# +
####
#### ...build model...
max_seq_length = 128

model, tokenizer = get_bert_tensorflow_hub_model(max_seq_length = max_seq_length
                                 , module_hub_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1")


# +
####
#### ...predict with before model...
df_words = bert_def[['id', 'w']].drop_duplicates(keep = 'first').reset_index(drop = True)
#df_words

lst_df_partition_output = list()


for index, part in df_words.iterrows(): 
    df_part = bert_def[(bert_def['id'] == part['id']) & (bert_def['w'] == part['w'])]
    #print(df_part)
    
    print('***************')
    sentence = list(df_part['token'])
    #print(sentence)
    
    sentence = ' '.join(sentence)
    print(sentence)
    
    
    print('***************')

    stokens = tokenizer.tokenize(sentence)
    #stokens = ["[CLS]"] + stokens + ["[SEP]"]

    input_ids = get_ids(stokens, tokenizer, max_seq_length)
    input_masks = get_masks(stokens, max_seq_length)
    input_segments = get_segments(stokens, max_seq_length)

    pool_embs, all_embs = model.predict([[np.asarray(input_ids)]
                                         , [np.asarray(input_masks)]
                                         , [np.asarray(input_segments)]])

    #print(input_ids)
    #print(input_masks)
    #print(input_segments)

    #print(all_embs.shape)
    #print(all_embs)
    print('*************************************')
    print('pool embedding of phrase \'{0}\''.format(str_test))
    #print(pool_embs.shape)
    #print(pool_embs)

    pool_embs = np.reshape(pool_embs[0], (1, bert_vector_dimension))[0]
    #print(pool_embs.shape)
    #print(pool_embs)
    
    
    #print( np.asarray(df_part[['id', 'w']].drop_duplicates(keep = 'first').reset_index(drop = True)) )
    row_result = np.reshape( np.append(np.asarray(df_part[['id', 'w']].drop_duplicates(keep = 'first').reset_index(drop = True))
                             , pool_embs)
                            , (1, 770) )
    
    print(row_result)
    print(row_result.shape)
    
    df_partition_output = pd.DataFrame(data = row_result
                                                , columns = ['id', 'w'] + bert_vector_colnames)

    lst_df_partition_output.append(df_partition_output)
    

####
#### ...build output dataframe...
df_output = pd.concat(lst_df_partition_output).reset_index(drop = True)

df_output
# -

df_output.to_csv(PATH_CHECKPOINT_BERT_WORDS_DEFINITION_POOLED_DIRECT
                 , sep='|'
                 , header=True, index=False, compression='gzip')



# ## LOAD BERT DATA - order reverse

# +
#start = time.time()
#os.remove(PATH_LOG_FILE)
#logger = create_logger(PATH_LOG_FILE)
#logger.info(' - starting execution')


#### LOAD DATASET
#### ...pairs of words with mannual similarities...
bert_def = pd.read_csv(PATH_CHECKPOINT_BERT_WORDS_DEFINITION_REORDERED_REVERSE, sep='|', header=0, compression='gzip')

####
#### DEFINE VECTOR REPRESENTATION COLNAMES
bert_vector_dimension = 768
#### ...get colnames with vector elements...
bert_vector_colnames = ['dim_{0}'.format(i) for i in range(1, bert_vector_dimension + 1)]


#### ...we get a few words to dev...
#bert_def = bert_def[bert_def.id.isin([428])]

bert_def.head(12)
# -

# ## BUILD AND USE MODEL - reverse order



# +
####
#### ...build model...
max_seq_length = 128

model, tokenizer = get_bert_tensorflow_hub_model(max_seq_length = max_seq_length
                                 , module_hub_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1")


# +
####
#### ...predict with before model...
df_words = bert_def[['id', 'w']].drop_duplicates(keep = 'first').reset_index(drop = True)
#df_words

lst_df_partition_output = list()


for index, part in df_words.iterrows(): 
    df_part = bert_def[(bert_def['id'] == part['id']) & (bert_def['w'] == part['w'])]
    #print(df_part)
    
    print('***************')
    sentence = list(df_part['token'])
    #print(sentence)
    
    sentence = ' '.join(sentence)
    print(sentence)
    
    
    print('***************')

    stokens = tokenizer.tokenize(sentence)
    #stokens = ["[CLS]"] + stokens + ["[SEP]"]

    input_ids = get_ids(stokens, tokenizer, max_seq_length)
    input_masks = get_masks(stokens, max_seq_length)
    input_segments = get_segments(stokens, max_seq_length)

    pool_embs, all_embs = model.predict([[np.asarray(input_ids)]
                                         , [np.asarray(input_masks)]
                                         , [np.asarray(input_segments)]])

    #print(input_ids)
    #print(input_masks)
    #print(input_segments)

    #print(all_embs.shape)
    #print(all_embs)
    print('*************************************')
    print('pool embedding of phrase \'{0}\''.format(str_test))
    #print(pool_embs.shape)
    #print(pool_embs)

    pool_embs = np.reshape(pool_embs[0], (1, bert_vector_dimension))[0]
    #print(pool_embs.shape)
    #print(pool_embs)
    
    
    #print( np.asarray(df_part[['id', 'w']].drop_duplicates(keep = 'first').reset_index(drop = True)) )
    row_result = np.reshape( np.append(np.asarray(df_part[['id', 'w']].drop_duplicates(keep = 'first').reset_index(drop = True))
                             , pool_embs)
                            , (1, 770) )
    
    print(row_result)
    print(row_result.shape)
    
    df_partition_output = pd.DataFrame(data = row_result
                                                , columns = ['id', 'w'] + bert_vector_colnames)

    lst_df_partition_output.append(df_partition_output)
    

####
#### ...build output dataframe...
df_output = pd.concat(lst_df_partition_output).reset_index(drop = True)

df_output
# -

df_output.to_csv(PATH_CHECKPOINT_BERT_WORDS_DEFINITION_POOLED_REVERSE
                 , sep='|'
                 , header=True, index=False, compression='gzip')
















