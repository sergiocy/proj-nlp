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
#### INPUT DATA FILES (ouput of reordering process)
PATH_CHECKPOINT_W2V_WORDS_DEFINITION_REORDERED_DIRECT = '../00data/nlp/tmp/ws353_w2v_reordered_direct_words_def_dict.csv.gz'
PATH_CHECKPOINT_W2V_WORDS_DEFINITION_REORDERED_REVERSE = '../00data/nlp/tmp/ws353_w2v_reordered_reverse_words_def_dict.csv.gz'
PATH_CHECKPOINT_BERT_WORDS_DEFINITION_REORDERED_DIRECT = '../00data/nlp/tmp/ws353_bert_reordered_direct_words_def_dict.csv.gz'
PATH_CHECKPOINT_BERT_WORDS_DEFINITION_REORDERED_REVERSE = '../00data/nlp/tmp/ws353_bert_reordered_reverse_words_def_dict.csv.gz'

#######################################3
#### DATASETS PROCESSED - CHECKPOINTS PATHS...
PATH_CHECKPOINT_W2V_WORDS_DEFINITION_COMPOSED_DIRECT_SUM = '../00data/nlp/tmp/ws353_w2v_composed_direct_words_def_dict_sum.csv.gz'
PATH_CHECKPOINT_W2V_WORDS_DEFINITION_COMPOSED_DIRECT_AVG = '../00data/nlp/tmp/ws353_w2v_composed_direct_words_def_dict_avg.csv.gz'
PATH_CHECKPOINT_W2V_WORDS_DEFINITION_COMPOSED_DIRECT_AVG_SEQ = '../00data/nlp/tmp/ws353_w2v_composed_direct_words_def_dict_avg_seq.csv.gz'
# -


def compound_vector_words_csv(logger = None
                               , df_input = None # input dataset with words and its definition as column way in dataframe
                               , cols_input_to_save = None # list with columns to save from input dataset
                               , col_partition = None
                               , cols_vector_representation = None
                               , vector_dimension = 300
                               , type_computation = 'sum'
                               , file_save_gz = None # file to save output
                               , sep_out = '|' # field separator in output data file
                               , verbose = True):


    try:
        
        ####
        #### ...list with different values in partition variable...
        elements_partition = df_input[col_partition].unique()
        
        ####
        #### ...define output variables...
        lst_df_partition_output = list()
        

        for part in elements_partition:
            #################################
            #### ...select partition associated with one word...
            df_part = df_input[df_input[col_partition]==part]
            
            if logger is not None:
                logger.info('reordering words in csv phrases')
                logger.info('input phrase: {}'.format(sentence))

                    
            #################################
            #### ...compute conposition. All operations get vector in defined input order
            #### - sum: vectors sum
            #### - avg: usual vector average
            #### - avg_sequence: average of each vector in sequence with average of before vectors
            values_part_to_save = np.asarray(df_part[cols_input_to_save].drop_duplicates(keep='first'))[0]
            #print(values_part_to_save)
            #print(values_part_to_save.shape)
            
            
            if type_computation == 'sum':
                #print(df_part)
                #print(df_partition_output)
                                
                for index, word_vector in df_part.iterrows():
                    #print(index)
                    if index == 0:
                        #### define the first vector to cumulate operation
                        cum_vector_operation = np.asarray(df_part.loc[index, cols_vector_representation])
                        #print(cum_vector_operation[0:5])
                        #print(cum_vector_operation.shape)
                        
                    else:
                        #### ...get the next vectors (after first)...
                        word_vector = np.asarray(df_part.loc[index, cols_vector_representation])
                        #print('vector to sum: {}'.format(word_vector[0:5]))
                        #print('vector to sum shape: {}'.format(word_vector.shape))
                        #### ...and compute the mean with before cummulate...
                        cum_vector_operation = np.add(cum_vector_operation, word_vector)
                        #print('vector result: {}'.format(cum_vector_operation[0:5]))
                        #print('vector result shape: {}'.format(cum_vector_operation.shape))
                        
                row_results = np.append(values_part_to_save, cum_vector_operation, axis = 0)
                #print('row result: {}'.format(row_results[0:5]))
                #print('row result shape: {}'.format(row_results.shape))  
                
            elif type_computation == 'avg':
                df_part_vectors = np.asarray(df_part[cols_vector_representation])
                row_results = np.mean(df_part_vectors, axis = 0)
                row_results = np.append(values_part_to_save, row_results, axis = 0)

            elif type_computation == 'avg_sequence':
                for index, word_vector in df_part.iterrows():
                    #print(index)
                    if index == 0:
                        #### define the first vector to cumulate operation
                        cum_vector_operation = np.asarray(df_part.loc[index, cols_vector_representation])
                        #print(cum_vector_operation[0:5])
                        #print(cum_vector_operation.shape)
                        
                    else:
                        #### ...get the next vectors (after first)...
                        word_vector = np.asarray(df_part.loc[index, cols_vector_representation])
                        #print('vector to sum: {}'.format(word_vector[0:5]))
                        #print('vector to sum shape: {}'.format(word_vector.shape))
                        #### ...and compute the mean with before cummulate...
                        cum_vector_operation = np.append(np.reshape(cum_vector_operation
                                                                    , (1, len(cols_vector_representation)))
                                                        , np.reshape(word_vector
                                                                    , (1, len(cols_vector_representation)))
                                                        , axis = 0)
                        cum_vector_operation = np.mean(cum_vector_operation, axis = 0)
                        #print(cum_vector_operation.shape)
                        #print('vector result: {}'.format(cum_vector_operation[0:5]))
                        #print('vector result shape: {}'.format(cum_vector_operation.shape))
                        
                row_results = np.append(values_part_to_save, cum_vector_operation, axis = 0)
                #print('row result: {}'.format(row_results[0:5]))
                #print('row result shape: {}'.format(row_results.shape)) 

            else:
                logger.info('this type_computation not exist')
                raise Exception

                
            ####
            #### ...define and build output dataframe...
            df_partition_output = pd.DataFrame(data = np.reshape(row_results, (1, len(row_results)))
                                                , columns = cols_input_to_save + cols_vector_representation)

            lst_df_partition_output.append(df_partition_output)
            
        
        ####
        #### ...build output dataframe...
        df_output = pd.concat(lst_df_partition_output).reset_index(drop = True)
        
        
        ####
        #### ...clean execution...
        del(lst_df_partition_output)
        
                
        if file_save_gz is not None:
            df_output.to_csv(file_save_gz, sep=sep_out, header=True, index=False, compression='gzip')


    except Exception:
        if logger is not None:
            logger.exception("ERROR computing")
        raise Exception


    return df_output




# ## LOAD W2V dictionary definitions reordered

# ### direct order
#
# in direct order we compute the three operations

# +
#start = time.time()
#os.remove(PATH_LOG_FILE)
#logger = create_logger(PATH_LOG_FILE)
#logger.info(' - starting execution')


#### LOAD DATASET
#### ...pairs of words with mannual similarities...
w2v_def = pd.read_csv(PATH_CHECKPOINT_W2V_WORDS_DEFINITION_REORDERED_DIRECT, sep='|', header=0, compression='gzip')

####
#### DEFINE VECTOR REPRESENTATION COLNAMES
w2v_vector_dimension = 300
#### ...get colnames with vector elements...
w2v_vector_colnames = ['dim_{0}'.format(i) for i in range(1, w2v_vector_dimension + 1)]


#### ...we get a few words to dev...
w2v_def = w2v_def[w2v_def.id.isin([1,2,3,4])]

w2v_def.head(12)
# +
w2v_composed_def_dict_sum = compound_vector_words_csv(logger = None
                               , df_input = w2v_def # input dataset with words and its definition as column way in dataframe
                               , cols_input_to_save = ['id', 'w'] # list with columns to save from input dataset
                               , col_partition = 'id'
                               , cols_vector_representation = w2v_vector_colnames # list with columns with vector representation
                               , vector_dimension = 300
                               , type_computation = 'sum'
                               , file_save_gz = None # file to save output
                               , sep_out = '|' # field separator in output data file
                               , verbose = True)

w2v_composed_def_dict_sum


# +
w2v_composed_def_dict_avg = compound_vector_words_csv(logger = None
                               , df_input = w2v_def # input dataset with words and its definition as column way in dataframe
                               , cols_input_to_save = ['id', 'w'] # list with columns to save from input dataset
                               , col_partition = 'id'
                               , cols_vector_representation = w2v_vector_colnames # list with columns with vector representation
                               , vector_dimension = 300
                               , type_computation = 'avg'
                               , file_save_gz = None # file to save output
                               , sep_out = '|' # field separator in output data file
                               , verbose = True)

w2v_composed_def_dict_avg
# +
w2v_composed_def_dict_avg_sequence = compound_vector_words_csv(logger = None
                               , df_input = w2v_def # input dataset with words and its definition as column way in dataframe
                               , cols_input_to_save = ['id', 'w'] # list with columns to save from input dataset
                               , col_partition = 'id'
                               , cols_vector_representation = w2v_vector_colnames # list with columns with vector representation
                               , vector_dimension = 300
                               , type_computation = 'avg_sequence'
                               , file_save_gz = None # file to save output
                               , sep_out = '|' # field separator in output data file
                               , verbose = True)

w2v_composed_def_dict_avg_sequence
# -

















