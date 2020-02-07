# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
#import nltk
from nltk.parse import CoreNLPParser





#### READING dataframe with phrases in one of its columns nd reorder to apply vector composition in this order 
def reorder_sentence_words_csv(logger = None
                               , df_input = None
                               , col_words_sentence = None
                               , type_order = 'syntactic'):

    try:
        logger.info('reordering words in csv phrases')



        #reorder_sentence_words()

        sentence = df_input[col_words_sentence].array        

        if type_order == 'syntactic':
            #sentence = " ".join(sentence)
            sentence = list(sentence)

            #parser = nltk.ChartParser(groucho_grammar)
            #for tree in parser.parse(sentence):
            #    print(tree)

            print(sentence)

            parser = CoreNLPParser(url='http://localhost:9000')
            p = list(parser.parse(sentence)) 

            print(type(p))
            print(p)
            #print(sentence)

            for tree in parser.parse(sentence):
                print(tree)
                

        elif type_order == 'direct':
            sentence = sentence
            print(sentence)

        elif type_order == 'reverse':
            sentence = list(sentence)
            sentence.reverse()
            
            sentence = np.asarray(sentence)
            print(sentence)

        else:
            logger.info('this type_order not exist')
            raise Exception


    except Exception:
        if logger is not None:
            logger.exception("ERROR reading csv")

        raise Exception

    #return df
