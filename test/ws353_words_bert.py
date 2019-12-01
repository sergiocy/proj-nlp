# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd


#from lib.py.datastructure.np_array_as_row_of_pd_df import np_array_as_row_of_pd_df
def np_array_as_row_of_pd_df(logger = None
                            , np_array = None
                            , pd_colnames_root = None):

    try:
        if pd_colnames_root is not None:
            df = pd.DataFrame(data=np_array.reshape((1, len(np_array)))
                      , columns=['{0}{1}'.format(pd_colnames_root, i) for i in range(1, len(np_array) + 1)]
                      )
        else:
            df = pd.DataFrame(data=np_array.reshape((1, len(np_array))))

    except Exception:
        if logger is not None:
            logger.exception("ERROR converting numpy array to pandas")
        else:
            print("ERROR converting numpy array to pandas")
        raise Exception

    return df







# +
PATH_CHECKPOINT_BERT_WORDS = '../../data/exchange/ws353_bert_words'
rep_w = pd.read_pickle(PATH_CHECKPOINT_BERT_WORDS)

rep_w.head(10)
rep_w.shape
# -

df_vec = pd.concat([np_array_as_row_of_pd_df(logger = None
                                                , np_array = rep_w['w_vect'][i]
                                                , pd_colnames_root = 'dim_') for i in range(len(rep_w.index))])
#new_index = list(range(len(df_vec.index)))
#print(new_index)
df_vec.index = rep_w.index
print(df_vec)

# +
#print(df_vec)

rep_w = pd.concat([rep_w[['id', 'w']], df_vec], axis = 1)

print(rep_w)
type(rep_w.index)
