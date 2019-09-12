# -*- coding: utf-8 -*-

import pandas as pd
import re
from nltk.tokenize import word_tokenize



def clean_saxon_genitive(token_in, logger=None):
    
    token = token_in

    try:
        print('-----------------')
        
    except Exception:
        if logger is not None:
            logger.exception("ERROR cleaning \',\' in token \'{0}\'".format(token_in))
        raise Exception
            
    return token


def clean_not_contraction(token_in, logger=None):
    
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

    clean_saxon_genitive(phrase)







