# -*- coding: utf-8 -*-

from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import PlaintextCorpusReader
import pandas as pd
import os
#import re




#### FUNCTION TO GET SOME COLUMNS (index_end_column) OF A CSV FILE (file) AS PANDAS-DF (colnames)
def format_content_file(file, colnames, index_end_column=7):
    f = open(file, encoding = 'utf8')
    lst_file = []
    
    for line in f:
        #### split by tabulations
        lst_line = line.split('\t')
        #### get first 7 elements (there is cases with complementary columns)
        lst_line = lst_line[0:index_end_column]
        lst_file.append(lst_line)
        
    df = pd.DataFrame(columns=colnames, data=lst_file)
    
    return df




def get_set_of_tokens(lst_sentences):
    set_words = []
    for r in lst_sentences:
        set_words.extend(list(r))
    set_words = sorted(set(set_words))
    
    return set_words


def get_corpus_and_ic(set_of_strings, file_corpus_output):
        
    if os.path.exists(file_corpus_output):
        os.remove(file_corpus_output)
        
    with open(file_corpus_output, "w+", encoding='utf8') as file_corpus:
        for token in set_of_strings:
            #file_corpus.write(token.replace('\n', '') + '\n')
            file_corpus.write(token + '\n')
    
    corpus = PlaintextCorpusReader(file_corpus_output.split('/')[0], file_corpus_output.split('/')[1])
    corpus_ic = wn.ic(corpus, False, 0.0)
    
    return corpus, corpus_ic


    
    

if __name__ == '__main__':

    print('in module input_processing')
    
 
