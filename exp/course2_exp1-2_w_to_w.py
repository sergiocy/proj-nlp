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

# # EXPERIMENT 1 and 2: single word VS single word (free and non-free context)

# +
from IPython.display import HTML
from IPython.display import display
import IPython.core.display as di # Example: di.display_html('<h3>%s:</h3>' % str, raw=

import os
import pandas as pd
import numpy as np
from numpy import array, dot, arccos, clip 
from numpy.linalg import norm 
from scipy.stats.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

di.display_html('<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>', raw=True)
di.display_html('''<button onclick="jQuery('.input_area').toggle(); jQuery('.prompt').toggle();">Toggle code</button>''', raw=True)
# +
##############################################
##############################################
#### FUNCTIONS
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
#### function to compute pearson coefficient between two variables
#### input: two numpy-arrays
def compute_pearson_coef(vector1, vector2):
    #### ...two ways to compute pearson coefficient...
    pearson1 = pearsonr(vector1, vector2)
    pearson2 = np.corrcoef(vector1, vector2)[0, 1]
    
    return pearson1, pearson2
##############################################
##############################################


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
PATH_CHECKPOINT_W2V_WORDS_CONTEXT = '../00data/nlp/tmp/ws353_w2v_words_context.csv.gz'
PATH_CHECKPOINT_BERT_WORDS = '../00data/nlp/tmp/ws353_bert_words.csv.gz'
PATH_CHECKPOINT_BERT_WORDS_CONTEXT = '../00data/nlp/tmp/ws353_bert_words_context.csv.gz'
#PATH_CHECKPOINT_BERT_WORDS_DEF_DICT = '../00data/nlp/tmp/ws353_bert_words_def_dict.csv.gz'
#PATH_CHECKPOINT_BERT_WORDS_DEF_WN = 'data/exchange/ws353_bert_def_wn'

####
#### ....we will save the dataset with pairs of words filtered; with associated representation...
PATH_OUTPUT_DATA_SIM = '../00data/nlp/tmp/ws353_input_sim_filtered_exp1_and_2.csv.gz'
# -

# ## Loading data
#
# We load datasets,
#
# - pairs of words with mannual similarity scores.
#
# - w2v embeddings
#
# - BERT embeddings

# +
#### DATASETS
#### ...pairs of words with mannual similarities...
df_pairs = pd.read_csv(PATH_INPUT_DATA_SIM, sep='|', header=0, compression='gzip')

#### ...free-context representations...
df_w2v_words = pd.read_csv(PATH_CHECKPOINT_W2V_WORDS, sep='|', header=0, compression='gzip')
df_bert_words = pd.read_csv(PATH_CHECKPOINT_BERT_WORDS, sep='|', header=0, compression='gzip')

#### ...free-context representations...
df_w2v_words_context = pd.read_csv(PATH_CHECKPOINT_W2V_WORDS_CONTEXT, sep='|', header=0, compression='gzip')
df_bert_words_context = pd.read_csv(PATH_CHECKPOINT_BERT_WORDS_CONTEXT, sep='|', header=0, compression='gzip')

#### PARAMETERS DEFINITION
#### w2v parameters
#### ...define dimensions in w2v representations...
w2v_vector_dimension = 300
#### ...get colnames with vector elements...
w2v_vector_colnames = ['dim_{0}'.format(i) for i in range(1, w2v_vector_dimension + 1)]

#### bert parameters
#### ...define dimensions in w2v representations...
bert_vector_dimension = 768
#### ...get colnames with vector elements...
bert_vector_colnames = ['dim_{0}'.format(i) for i in range(1, bert_vector_dimension + 1)]
# -

print(df_pairs.shape)
df_pairs = df_pairs.dropna()
print(df_pairs.shape)
df_pairs.head()

print(df_w2v_words.shape)
df_w2v_words = df_w2v_words.dropna()
print(df_w2v_words.shape)
df_w2v_words.head()

print(df_bert_words.shape)
df_bert_words = df_bert_words.dropna()
print(df_bert_words.shape)
df_bert_words.head()

# For this pairs we will take the vector representations of each one and we will compute the similarities, using cosine metrics, and we compare them with mannual scores in the above table. 
#
#
# ### Filter data
#
# Firstly we filter pairs of words taking pairs that exists in vector representations datasets (w2v and BERT).

# +
lst_words_bert_rep = df_bert_words['w'].unique()
lst_words_w2v_rep = df_w2v_words['w'].unique()
lst_words_bert_context = df_bert_words_context[df_bert_words_context['w'] == df_bert_words_context['token']].w.unique()
lst_words_w2v_context = df_w2v_words_context[df_w2v_words_context['w'] == df_w2v_words_context['token']].w.unique()

df_pairs = df_pairs[#### ...filter words with w2v-representations free-context...
                    (df_pairs['w1'].isin(lst_words_w2v_rep)) & (df_pairs['w2'].isin(lst_words_w2v_rep)) &
                    #### ...filter words with BERT-representations free-context...
                    (df_pairs['w1'].isin(lst_words_bert_rep)) & (df_pairs['w2'].isin(lst_words_bert_rep)) &
                    #### ...filter words with w2v-representations non-free-context...
                    (df_pairs['w1'].isin(lst_words_w2v_context)) & (df_pairs['w2'].isin(lst_words_w2v_context)) &
                    #### ...filter words with BERT-representations non-free-context...
                    (df_pairs['w1'].isin(lst_words_bert_context)) & (df_pairs['w2'].isin(lst_words_bert_context))]
                     
print(df_pairs.shape)
df_pairs.head()
# -

# In this way we had 353 pairs of that that are reducced to 342 (some words has not representations; "jerusalem", in example).

# ## Similarities computation (free context)

# +
#### ...get pair of words cosine similarities...
lst_w2v_sim_cosine = list()
lst_bert_sim_cosine = list()

for index, row in df_pairs.iterrows():
    #one_pair_words = df_pairs.loc[0]
    w1 = row['w1'] #one_pair_words.w1
    w2 = row['w2'] #one_pair_words.w2
    #one_pair_words
    #print('words: {0} and {1}'.format(w1, w2))
    
    #### ...w2v-representations cosine...
    vector1 = np.asarray(df_w2v_words[df_w2v_words['w'] == w1][w2v_vector_colnames])[0]
    vector2 = np.asarray(df_w2v_words[df_w2v_words['w'] == w2][w2v_vector_colnames])[0]
    #print(vector1[1:3])
    #print(vector2[1:3])
    magn, cosine, angle = compute_similarity_cosin(vector1, vector2)
    lst_w2v_sim_cosine = lst_w2v_sim_cosine + [cosine]
    #print('row {0} - words: {1} and {2} - cosine {3}'.format(index, w1, w2, cosine))
    
    #### ...bert-representations cosine...
    vector1 = np.asarray(df_bert_words[df_bert_words['w'] == w1][bert_vector_colnames])[0]
    vector2 = np.asarray(df_bert_words[df_bert_words['w'] == w2][bert_vector_colnames])[0]
    #print(vector1[1:3])
    #print(vector2[1:3])
    magn, cosine, angle = compute_similarity_cosin(vector1, vector2)
    lst_bert_sim_cosine = lst_bert_sim_cosine + [cosine]
    #print('row {0} - words: {1} and {2} - cosine {3}'.format(index, w1, w2, cosine))
    
df_pairs['cos_w2v'] = lst_w2v_sim_cosine
df_pairs['cos_bert'] = lst_bert_sim_cosine
df_pairs.head()
# -

# ## Similarities computation (non-free context)
#
# To consider words in context and review the influence of that, we generate datasets in the next way.
#
# For w2v representations, we have,

#### ...w2v-representations
df_w2v_words_context.head(10)

# and for BERT representations, we have,

df_bert_words_context.head(10) 

# where we have a phrase that contains the word (column "token" read in vertical way).
#
# **NOTE**: remark that gensim package, used to get the w2v representations, remove some stop words by default. So, in w2v representations we can observe that word "a" in phrase associated to word "tiger", have been removed in w2v representations. 
#
# From this datasets, we can select the representations associated for each word in context,

print(df_pairs.shape)

df_w2v_words_context = df_w2v_words_context[df_w2v_words_context['w'] == df_w2v_words_context['token']]
print(df_w2v_words_context.shape)
df_w2v_words_context.head()

df_bert_words_context = df_bert_words_context[df_bert_words_context['w'] == df_bert_words_context['token']]
print(df_bert_words_context.shape)
df_bert_words_context.head()

# +
#### ...get pair of words cosine similarities...
lst_w2v_sim_cosine = list()
lst_bert_sim_cosine = list()

for index, row in df_pairs.iterrows():
    #one_pair_words = df_pairs.loc[0]
    w1 = row['w1'] #one_pair_words.w1
    w2 = row['w2'] #one_pair_words.w2
    #one_pair_words
    #print('words: {0} and {1}'.format(w1, w2))
    
    #### ...w2v-representations cosine...
    vector1 = np.asarray(df_w2v_words_context[df_w2v_words_context['w'] == w1][w2v_vector_colnames])[0]
    vector2 = np.asarray(df_w2v_words_context[df_w2v_words_context['w'] == w2][w2v_vector_colnames])[0]
    #print(vector1[1:3])
    #print(vector2[1:3])
    magn, cosine, angle = compute_similarity_cosin(vector1, vector2)
    lst_w2v_sim_cosine = lst_w2v_sim_cosine + [cosine]
    #print('row {0} - words: {1} and {2} - cosine {3}'.format(index, w1, w2, cosine))
    
    #### ...bert-representations cosine...
    vector1 = np.asarray(df_bert_words_context[df_bert_words_context['w'] == w1][bert_vector_colnames])[0]
    vector2 = np.asarray(df_bert_words_context[df_bert_words_context['w'] == w2][bert_vector_colnames])[0]
    #print(vector1[1:3])
    #print(vector2[1:3])
    magn, cosine, angle = compute_similarity_cosin(vector1, vector2)
    lst_bert_sim_cosine = lst_bert_sim_cosine + [cosine]
    #print('row {0} - words: {1} and {2} - cosine {3}'.format(index, w1, w2, cosine))
    
df_pairs['cos_w2v_context'] = lst_w2v_sim_cosine
df_pairs['cos_bert_context'] = lst_bert_sim_cosine
df_pairs.head()

# +
####
#### ...we will save this dataset and we will take as reference in next experiments. 
#### In this way we warranty that use the same number of pairs of words
# df_pairs.to_csv(PATH_OUTPUT_DATA_SIM, sep = '|', header = True, compression = 'gzip')
# -

# ## Results
#
# We compute **correlation** (Pearson coefficiente) for each case. So,

pd.DataFrame({'w2v': [compute_pearson_coef(df_pairs['cos_w2v'], df_pairs['score'])[1] ,
                      compute_pearson_coef(df_pairs['cos_w2v_context'], df_pairs['score'])[1]],
              'bert': [compute_pearson_coef(df_pairs['cos_bert'], df_pairs['score'])[1] ,
                      compute_pearson_coef(df_pairs['cos_bert_context'], df_pairs['score'])[1]]
                      }, index = ['free-context', 'non-free-context'])

# and we can do some comments and observations:
#
# - context does not influence w2v representations. So, we get the expected behaviour, because w2v model does not take in account it.
#
# - In the case of BERT model, context increase correlation. It is the expected behaviour.
#
# - Anyway, w2v representations get a best correlation.

# and we review the cosine distributions (KDE estimations),

# +
plt.figure(figsize = (15,10))

sns.distplot(df_pairs['cos_w2v']
             , hist=False
             , kde=True 
             #, bins=int(180/5)
             , color = 'darkblue' 
             #hist_kws={'edgecolor':'black'},
             #kde_kws={'linewidth': 4}
             , label = 'w2v'
            )

sns.distplot(df_pairs['cos_bert']
             , hist=False
             , kde=True 
             #, bins=int(180/5)
             , color = 'green' 
             #hist_kws={'edgecolor':'black'},
             #kde_kws={'linewidth': 4}
             , label = 'bert'
            )

sns.distplot(df_pairs['cos_bert_context']
             , hist=False
             , kde=True 
             #, bins=int(180/5)
             , color = 'red' 
             #hist_kws={'edgecolor':'black'},
             #kde_kws={'linewidth': 4}
             , label = 'bert_context'
            )
plt.xlabel('cos')
plt.show()
# -



# ## Next steps from results
#
# - Space topology: the shape of words clusters
#
# - high/low cosine depends on gramatical categories
#
# - review cosine values/words greater than 1 and lower than 0. That imply angles out of range 0-90 degrees






