# -*- coding: utf-8 -*-

import pandas as pd
import re
from nltk.tokenize import word_tokenize


#### CLEAN TEXT IN COLUMN "col" OF DF-PANDAS

def clean_text_column(df
                    , col
                    , del_punct=False
                    , lst_punt_to_del=[]
                    , lcase=False
                    , tokenize=False
                    , del_saxon_genitive=False
                    , not_contraction=False
                    , percentage=False
                    , exist_float_number=False
                    , logger=None):
   
    try:
        if lcase: df[col] = df[col].apply(lambda phrase: str(phrase).lower())
        if del_saxon_genitive: df[col] = df[col].apply(lambda phrase: re.sub(r'(\'s)', '', phrase))
        if not_contraction: df[col] = df[col].apply(lambda phrase: re.sub(r'(n\'t)', ' not', phrase))
        if percentage: df[col] = df[col].apply(lambda phrase: re.sub(r'%', ' percent', phrase))
        if del_punct:
            for punct in lst_punt_to_del:
                if exist_float_number:
                    if punct==',':
                        df[col] = df[col].apply(lambda phrase: re.sub(r',[^0-9]', '', phrase))
                    else:
                        df[col] = df[col].apply(lambda phrase: re.sub(r'{0}'.format(str(punct)), '', phrase))
                else:
                    df[col] = df[col].apply(lambda phrase: re.sub(r'{0}'.format(str(punct)), '', phrase))
        if tokenize: df[col] = df[col].apply(lambda phrase: word_tokenize(phrase))

        if logger is not None:
            logger.info(' - column \'{0}\' in pandas dataframe cleaned; first rows...'.format(col, df))
            logger.info('\n{0}'.format(df.loc[0:4]))

    except Exception:
        if logger is not None:
            logger.exception("ERROR cleaning pandas column")
        raise Exception
            
    return df






