# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
#import re
#import nltk
from nltk.parse import CoreNLPParser
import nltk.parse.api
#import nltk

from app.service.text.reorder_syntactic_tokenized_sentence import *
from app.service.text.reorder_syntactic_tokenized_sentence_regex import *




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
                               , file_save_gz = None # file to save output
                               , sep_out = '|' # field separator in output data file
                               , verbose = True):


    try:

        #lst_ordered_words = list()
        lst_df_reordered = list()
        elements_partition = df_input[col_partition].unique()

        for part in elements_partition:
            #################################
            #### ...select partition associated with one word...
            df_part = df_input[df_input[col_partition]==part]
            #### ...to optimize memory, we delete this word from input dataframe...
            #df_input = df_input[df_input[col_id_words_sentence] != int(df_part[col_id_words_sentence].unique())]
            #print('rows in input dataframe: {}'.format(df_input.shape))

            ####
            #### ...we get words sentence and words ids as lists...
            sentence = df_part[col_words_sentence].array
            sentence = list(sentence)
            id_token = list(df_part[col_id_words_sentence].array)

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

            else:
                logger.info('this type_order not exist')
                raise Exception




            #################################
            #### ...we build the output as list of phrases with reordered words...
            lst_reordered_token = sentence
            lst_reordered_id_token = list()

            #print(sentence)
            #print(paired_token)
            #print('----------------')

            for reordered_token in lst_reordered_token:
                #print()
                for i_pair in range(len(paired_token)):
                    if reordered_token == paired_token[i_pair][0]:
                        lst_reordered_id_token = lst_reordered_id_token + [paired_token[i_pair][1]]
                        del(paired_token[i_pair])
                        #print(paired_token)
                        break

            #print('----------------')
            #print(lst_reordered_token)
            #print(lst_reordered_id_token)



            #################################
            #### ...we add to input dataset as new column...
            #print('adding new columns')
            #df_input[col_words_sentence_reordered] = np.asarray(lst_reordered_token)
            #df_input[col_id_words_sentence_reordered] = np.asarray(lst_reordered_id_token)

            ####
            #### ...generate dataset as input dataframe with reordered words, ids and vectors...
            df_reordered = pd.DataFrame(data = df_part[cols_input_to_save])
            df_reordered[col_id_words_sentence] = np.asarray(lst_reordered_id_token)
            df_reordered[col_words_sentence] = np.asarray(lst_reordered_token)

            #print(df_reordered)
            #print(df_part)

            #### ...add vectors...
            #df_reordered = df_reordered.merge(df_part[cols_vector], left_on=[col_id_words_sentence_reordered, col_words_sentence_reordered]
            #                           , right_on=[col_id_words_sentence, col_words_sentence])

            df_reordered = pd.merge(df_reordered#.reset_index(drop = True)
                                    , df_part#[[cols_vector]]#.reset_index(drop = True)
                                    , left_on=[col_id_words_sentence, col_words_sentence]
                                    , right_on=[col_id_words_sentence, col_words_sentence]
                                    , how = 'left'
                                    , suffixes = ('', '_y'))
            ####
            #### ...we remove variables duplicated by join...
            df_reordered = df_reordered.drop(['id_y', 'w_y'], axis = 1)
            print(df_reordered)

            ####
            #### ...redefine the reordered dataframe as data in input dataframe (substitute the reordered data)
            lst_df_reordered.append(df_reordered)

            ####
            #### ...clean data...
            del(df_reordered)
            del(df_part)


        ####
        #### ...we build output dataframe with reordered words...
        df_output = pd.concat(lst_df_reordered)

        if file_save_gz is not None:
            df_output.to_csv(file_save_gz, sep=sep_out, header=True, index=False, compression='gzip')


    except Exception:
        if logger is not None:
            logger.exception("ERROR reading csv")
        raise Exception


    return df_output
