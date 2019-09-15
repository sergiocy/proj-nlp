# -*- coding: utf-8 -*-

import pandas as pd
import re
from nltk.tokenize import word_tokenize

from .clean_token_punctuation import clean_punctuation_in_token
from .clean_token_en_expression import clean_en_expression_in_token



#### CLEAN ENGLISH TOKEN
def clean_english_token(token_in
                        , lcase=False
                        , lst_punct_to_del = []
                        , lst_en_exp_to_del = []
                        , logging_tokens_cleaning = True
                        , logger=None):
    
    try:
        token = token_in.strip()


        if lcase: 
            token = str(token).lower()

        if len(lst_punct_to_del)>0:
            for punct in lst_punct_to_del:
                token = clean_punctuation_in_token(token, punct, logger=logger)

        if len(lst_en_exp_to_del)>0:
            for exp in lst_en_exp_to_del:
                token = clean_punctuation_in_token(token, exp, logger=logger)


        token = token.strip()

        if logger is not None and logging_tokens_cleaning:
            logger.info(' - cleaning token \'{0}\''.format(token_in))
            logger.info('\n -->> \'{0}\' -->> \'{1}\''.format(token_in, token))    

        
    except Exception:
        if logger is not None:
            logger.exception("ERROR cleaning token \'{0}\'".format(token_in))
        raise Exception

    return token




#### CLEAN PHRASE
#### NOTE: we must do it for works with only one token
def clean_phrase(str_phrase_in
                , language = 'en'
                , lcase=False
                , lst_punct_to_del = []
                , lst_en_exp_to_del = []
                , tokenized=False
                , logging_tokens_cleaning = True
                , logger=None):
   
    
    try:
        if logger is not None:
            logger.info(' - cleaning phrase \'{0}\''.format(str_phrase_in))
        
        phrase = str_phrase_in.strip()
        phrase = word_tokenize(phrase)

        if language == 'en':
            phrase = [clean_english_token(token
                                            , lcase=lcase
                                            , lst_punct_to_del = lst_punct_to_del
                                            , lst_en_exp_to_del = lst_en_exp_to_del
                                            , logging_tokens_cleaning = logging_tokens_cleaning 
                                            , logger=logger) for token in phrase]

            phrase = [token for token in phrase if token != '']
            
            if not tokenized:
                phrase = ' '.join(phrase)


        if logger is not None:
            logger.info('\n -->> \'{0}\' -->> \'{1}\''.format(str_phrase_in, phrase))
        
    except Exception:
        if logger is not None:
            logger.exception("ERROR cleaning phrase \'{0}\'".format(str_phrase_in))
        raise Exception
            
    return phrase




if __name__ == '__main__':
    phrase1 = 'To Like something'
    phrase2 = 'To Like something'

    clean_phrase(phrase)







