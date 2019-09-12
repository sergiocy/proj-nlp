# -*- coding: utf-8 -*-

import pandas as pd
import re
from nltk.tokenize import word_tokenize


#### CLEAN PHRASE
def clean_comma_in_token(token_in, logger=None):
    
    token = token_in

    try:
        print('-----------------')
        
    except Exception:
        if logger is not None:
            logger.exception("ERROR cleaning \',\' in token \'{0}\'".format(token_in))
        raise Exception
            
    return token


if __name__ == '__main__':
    phrase = 'To Like something'

    clean_comma_in_token(phrase)







