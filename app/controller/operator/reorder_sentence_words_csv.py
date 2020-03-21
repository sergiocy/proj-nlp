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

                #### TODO: code to up standford server API
                #### ...up syntactical parsin standford API..
                # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse,depparse -status_port 9000 -port 9000 -timeout 15000 &

                #sentence = reorder_syntactic_tokenized_sentence(logger = logger
                #                                                , lst_sentence = sentence
                #                                                , use_stanford_parser = True
                #                                                , verbose = verbose)

                sentence = reorder_syntactic_tokenized_sentence_regex(logger = logger
                                                                        , lst_sentence = sentence
                                                                        , use_stanford_parser = True
                                                                        , verbose = False)

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
        df_input[col_words_sentence_reordered] = np.asarray(lst_ordered_words)


    except Exception:
        if logger is not None:
            logger.exception("ERROR reading csv")
        raise Exception


    return df_input
