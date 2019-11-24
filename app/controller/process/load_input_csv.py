# -*- coding: utf-8 -*-

import pandas as pd


#### READING CSV FILE INPUT
def read_csv(logger = None
                , new_colnames = None
                , file_input = None
                , has_header = True
                , sep = ';'
                , encoding = 'utf-8'
                , has_complete_rows = True):

    try:
        if logger is not None:
            logger.info(' - read file {0}'.format(file_input))

        if has_header:
            df = pd.read_csv(file_input, sep=sep, encoding=encoding)
            if new_colnames is not None:
                df.columns = new_colnames
            if has_complete_rows:
                df = df.dropna()
        else:
            df = pd.read_csv(file_input, sep=sep, encoding=encoding, header=None)
            if new_colnames is None and logger is not None:
                logger.warn(" - dataframe without header")
            else: 
                df.columns = new_colnames
            if has_complete_rows:
                df = df.dropna()


        if logger is not None:
            logger.info(' - csv loaded as pandas dataframe; first rows...')
            logger.info('\n{0}'.format(df.loc[0:4]))
            logger.info(' - column names: {0}'.format(df.columns))
            logger.info(' - dataframe dimensions: {0}'.format(df.shape))

    except Exception:
        if logger is not None:
            logger.exception("ERROR reading csv")
        
        raise Exception

    return df

