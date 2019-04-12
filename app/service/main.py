# -*- coding: utf-8 -*-



from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd

from input_text.input_processing import *
from vectorizer.liu_vectorizer import *
from similarity.computing_similarity import *

from similarity.computing_similarity import *




def prepare_input(file_csv, colnames_csv, col_text1, col_text2, file_corpus_output_ic):

    df_test = format_content_file(file_csv, colnames)
    #### select a few rows to developement
    df_test = df_test.iloc[0:100, :]
    
    #### LOAD INFORMATION CONTENT FROM CORPUS
    lst_sentences = list(np.append(df_test[col_text1], df_test[col_text2]))
    #### ...we get the corpus used to compute IC in a file
    full_corpus, corpus_ic = get_corpus_and_ic(set_of_strings=lst_sentences, file_corpus_output=file_corpus_output_ic)
    
    #### CLEAN AND TOKENIZE TEXT 
    df_test = clean_text(df_test, col_text1)
    df_test = clean_text(df_test, col_text2)
    
    return df_test, corpus_ic
    
    
    
def process_liu(sen1, sen2):
    print('launching process')

    
                
       
    

if __name__ == '__main__':

    #### LOAD AND CLEANING DATA    
    #file_dev = 'corpus/sts-dev.csv'
    #file_train = 'corpus/sts-train.csv'
    file_test = 'corpus/sts-test.csv' 
    file_corpus_output = 'corpus/corpus_tmp.txt'
    colnames = ['genre', 'filename', 'year', 'number', 'score', 'sentence1', 'sentence2']
    
    df_text, ic = prepare_input(file_test, colnames, 'sentence1', 'sentence2', file_corpus_output)
       
    ####
    #### TESTING WORD SIMILARITY
    #print(get_words_similarity('girl', 'hair', type_score = 'path_similarity', corpus_ic = corpus_ic))
    
    print('###############')
    frase1 = df_text.loc[0]['sentence1']
    frase2 = df_text.loc[0]['sentence2']
    #frase1 = ["consumer","would","still","have","to","get","a","descramble","security","card","from","their","cable","operator","to","plug","into","the","set"]
    #frase2 = ["to","watch","pay","television","consumer","would","insert","into","the","set","a","security","card","provide","by","their","cable","service"]
    #get_vector_words_appearence(frase1, frase2)
    
    print('###############')
    vector1, vector2 = get_vector_representation(frase1, frase2, type_score='lin', corpus_ic=ic)
    
    magn, cos, ang = compute_similarity_cosin(vector1, vector2)
    ang = (360*ang)/(2*3.1416)
    print(magn)
    print(cos)
    print(ang)
    
 
