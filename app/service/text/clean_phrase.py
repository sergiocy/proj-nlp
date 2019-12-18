# -*- coding: utf-8 -*-

import pandas as pd
import re
from nltk.tokenize import word_tokenize

from lib.py.nlp.cleaner.clean_english_token import clean_english_token



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

        print(phrase)

        if language == 'en':
            phrase = [clean_english_token(token
                                            , lcase = lcase
                                            , lst_punct_to_del = lst_punct_to_del
                                            , lst_en_exp_to_del = lst_en_exp_to_del
                                            , logging_tokens_cleaning = False
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

    clean_phrase(phrase1)
