# -*- coding: utf-8 -*-

import pandas as pd


#### READING CSV FILE INPUT
def read_csv_and_add_or_change_colnames(new_colnames = None
                                        , logger = None
                                        , file_input = None
                                        , header = True
                                        , sep = ';'
                                        , encoding = 'utf-8'):

    try:
        if logger is not None:
            logger.info(' - read file {0}'.format(file_input))

        if header:
            df = pd.read_csv(file_input, sep=sep, encoding=encoding)
            if new_colnames is not None:
                df.columns = new_colnames
        else:
            df = pd.read_csv(file_input, sep=sep, encoding=encoding, header=None)
            if new_colnames is None and logger is not None:
                logger.warn(" - dataframe without header")
            else: 
                df.columns = new_colnames

        if logger is not None:
            logger.info(' - csv loaded as pandas dataframe; first rows...')
            logger.info('\n{0}'.format(df.loc[0:4]))

    except Exception:
        if logger is not None:
            logger.exception("ERROR reading csv")
        
        raise Exception

    return df

