# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # EXPERIMENT 1: single word VS single word

# +
from IPython.display import HTML
from IPython.display import display
import IPython.core.display as di # Example: di.display_html('<h3>%s:</h3>' % str, raw=

import pandas as pd
import numpy as np
from numpy import array, dot, arccos, clip 
from numpy.linalg import norm 



di.display_html('<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>', raw=True)
di.display_html('''<button onclick="jQuery('.input_area').toggle(); jQuery('.prompt').toggle();">Toggle code</button>''', raw=True)
# -





# +
####
#### ...execution files...
#PATH_LOG_FILE = 'log/log.log'

####
#### CSV FILES - DATA INPUT
PATH_INPUT_DATA_SIM = '../00data/nlp/tmp/ws353_input_sim.csv.gz'


####
#### data files during execution / checkpoints
PATH_CHECKPOINT_W2V_WORDS = '../00data/nlp/tmp/ws353_w2v_words.csv.gz'
#PATH_CHECKPOINT_BERT_WORDS_CONTEXT = '../00data/nlp/tmp/ws353_bert_words_context.csv.gz'
#PATH_CHECKPOINT_BERT_WORDS_DEF_DICT = '../00data/nlp/tmp/ws353_bert_words_def_dict.csv.gz'
#PATH_CHECKPOINT_BERT_WORDS_DEF_WN = 'data/exchange/ws353_bert_def_wn'
# -

# We load pairs of words with mannual similarity scores.

df_pairs = pd.read_csv(PATH_INPUT_DATA_SIM, sep='|', header=0, compression='gzip')

print(df_pairs.shape)
df_pairs = df_pairs.dropna()
print(df_pairs.shape)
df_pairs.head(15)

# For this pairs we will take the vector representations of each one and we will compute the similarities, using cosine and ICM metrics, and we compare them with mannual scores in the above table. 

# ## W2V representations

# +
w2v_vector_dimension = 300

df_w2v_words = pd.read_csv(PATH_CHECKPOINT_W2V_WORDS, sep='|', header=0, compression='gzip')

print(len(df_w2v_words.w.unique()))
print(df_w2v_words.shape)

df_w2v_words.head()
# -



# +
####
#### similarity metric based on cosin between vector (for text vectorial representations)  
#### numpy arrays as input arguments
def compute_similarity_cosin(vector1, vector2):
    u = vector1 #array(vector1) 
    v = vector2 #array(vector2) 
    d = dot(u, v)
    c = dot(u,v)/norm(u)/norm(v) # -> cosine of the angle 
    angle = arccos(clip(c, -1, 1)) # if you really want the angle 

    return d, c, angle  

####
#### similarity metric based on cosin between vector (for text vectorial representations)  
#### numpy arrays as input arguments
def compute_similarity_icm(vector1, vector2):
    print('compute ICM')
# -



# +
#### ...get colnames with vector elements...
w2v_vector_colnames = ['dim_{0}'.format(i) for i in range(1, w2v_vector_dimension + 1)]

#### ...get pair of words by row...
one_pair_words = df_pairs.loc[0]
w1 = one_pair_words.w1
w2 = one_pair_words.w2

magn, cosine, angle = compute_similarity_cosin(np.asarray(df_w2v_words[df_w2v_words['w'] == w1][w2v_vector_colnames])[0]
                                                         , np.asarray(df_w2v_words[df_w2v_words['w'] == w2][w2v_vector_colnames])[0])

one_pair_words
# -

[]










