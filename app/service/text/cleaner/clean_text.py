# -*- coding: utf-8 -*-

import pandas as pd
import re
from nltk.tokenize import word_tokenize



def clean_english_token(token_in
                , lcase=True
                , logger=None):
    
    token = token_in
    
    try:
        print('-----------------')
        print(phrase)
        phrase = word_tokenize(phrase)
        print(phrase)

        if lcase: 
            phrase = str(phrase).lower()

        print(phrase)
        
    except Exception:
        if logger is not None:
            logger.exception("ERROR cleaning token \'{0}\'".format(token_in))
        raise Exception

    return token



#### CLEAN PHRASE
#### NOTE: we must do it for works with only one token
def clean_phrase(phrase_in
                , language = 'en'
                , lcase=True
                , tokenized=False
                , logger=None):
   
    phrase = phrase_in

    try:
        print('-----------------')
        print(phrase)
        if language == 'en':
            phrase = word_tokenize(phrase)
            phrase = [clean_english_token(token) for token in phrase]
        print(phrase)

        if not tokenized:
            phrase = ' '.join(phrase)

        print(phrase)
        print('-----------------')
        
    except Exception:
        if logger is not None:
            logger.exception("ERROR cleaning phrase \'{0}\'".format(phrase_in))
        raise Exception
            
    return phrase


if __name__ == '__main__':
    phrase = 'To Like something'
    phrase = 'To Like something'

    clean_phrase(phrase)






