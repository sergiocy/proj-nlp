# -*- coding: utf-8 -*-



#from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd

from input_text.input_processing import *
import vectorizer.liu_vectorizer as liu
#import vectorizer.agirre_vectorizer as agirre
from similarity.similarity_metric import *





def prepare_input(file_csv, colnames_csv, col_text1, col_text2, file_corpus_output_ic):

    df_test = format_content_file(file_csv, colnames)
    
    #### LOAD INFORMATION CONTENT FROM CORPUS
    lst_sentences = list(np.append(df_test[col_text1], df_test[col_text2]))
    #### ...we get the corpus used to compute IC in a file
    full_corpus, corpus_ic = get_corpus_and_ic(set_of_strings=lst_sentences, file_corpus_output=file_corpus_output_ic)
    
    #### CLEAN AND TOKENIZE TEXT 
    df_test = clean_text(df_test, col_text1)
    df_test = clean_text(df_test, col_text2)
    
    return df_test, corpus_ic
    
    

def process_liu(sen1, sen2, ic, type_similarity = 'res'):
    print('LAUNCHING LIU PROCESS WITH SIMILARITY {0}'.format(type_similarity))
    print(sen1)
    vector1, vector2 = liu.get_vector_representation(sen1, sen2, type_score=type_similarity, corpus_ic=ic)
    magn, cos, ang = compute_similarity_cosin(vector1, vector2)
    ang = (360*ang)/(2*3.1416)
    
    #print(vector1)
    #print(vector2)
    #print(magn)
    #print(cos)
    #print(ang)
    
    return vector1, vector2, magn, cos, ang



def process_agirre(sen1, sen2):
    print('LAUNCHING AGIRRE PROCESS WITH SIMILARITY')
    #vector1, vector2 = liu.get_vector_representation(sen1, sen2, type_score=type_similarity, corpus_ic=ic)
    
    n_overlapped = get_number_words_overlapped(sen1, sen2)
    agirre_metric = (2*n_overlapped)/(len(sen1)+len(sen2))
        
    lst_syn_overlapped, n_syn_overlapped = get_number_synsets_overlapped(sen1, sen2)
    agirre_metric_synsets = (2*n_syn_overlapped)/(len(sen1)+len(sen2))
    
    return agirre_metric, agirre_metric_synsets
    


def run_dataframe_sentences(df, field_text1, field_text2, ic, file_output_dataframe):
    print('RUNNING PROCESS...')
        
    #vector1, vector2, magn, cos, ang = process_liu(frase1[0], frase1[0], ic, type_similarity = 'path_similarity')
    #print(cos)
    #print( process_liu(frase1[0], frase1[0], ic, type_similarity = 'path_similarity')[0] )
    
    df['liu_path'] = df.apply(lambda x: process_liu(list(x[field_text1]), list(x[field_text2]), ic, type_similarity = 'path_similarity')[3], axis=1) 
    df['liu_res'] = df.apply(lambda x: process_liu(list(x[field_text1]), list(x[field_text2]), ic, type_similarity = 'res')[3], axis=1) 
    df['liu_lin'] = df.apply(lambda x: process_liu(list(x[field_text1]), list(x[field_text2]), ic, type_similarity = 'lin')[3], axis=1)
    df['liu_jcn'] = df.apply(lambda x: process_liu(list(x[field_text1]), list(x[field_text2]), ic, type_similarity = 'jcn')[3], axis=1) 
    
    df['agirre_words'] = df.apply(lambda x: process_agirre(list(x[field_text1]), list(x[field_text2]))[0], axis=1) 
    df['agirre_synsets'] = df.apply(lambda x: process_agirre(list(x[field_text1]), list(x[field_text2]))[1], axis=1)
    #df['agirre_words'], df['agirre_synsets'] = df.apply(lambda x: process_agirre(list(x[field_text1]), list(x[field_text2])), axis=1)
    #df['agirre_words', 'agirre_synsets'] = df.apply(lambda x: list( process_agirre(list(x[field_text1]), list(x[field_text2])) ), axis=1)
    #df[['agirre_words', 'agirre_synsets']] = zip(*df.map(process_agirre(list(df[field_text1]), list(df[field_text2]))))
    #df['agirre_words', 'agirre_synsets'] = df.apply(lambda x: list(process_agirre(list(x[field_text1]), list(x[field_text2]))))
    
    #df['agirre_words'] = map(lambda x: list(process_agirre(list(x[field_text1]), list(x[field_text2]))[0] ), df[field_text1] )
    #print(df)
    
    df.to_csv(file_output_dataframe, sep='~')
    
    return df
    
               
       

    

if __name__ == '__main__':

    #### LOAD AND CLEANING DATA    
    #file_dev = 'corpus/sts-dev.csv'
    #file_train = 'corpus/sts-train.csv'
    file_test = 'corpus/sts-test.csv' 
    colnames = ['genre', 'filename', 'year', 'number', 'score', 'sentence1', 'sentence2']
    
    file_corpus_output = 'corpus/corpus_to_compute_ic.txt'
    
    
    df_text, ic = prepare_input(file_test, colnames, 'sentence1', 'sentence2', file_corpus_output)
       
    ####
    #### TESTING WORD SIMILARITY
    #print(get_words_similarity('girl', 'hair', type_score = 'path_similarity', corpus_ic = corpus_ic))
    
    ####select a few rows to developement
    #df_text = df_text.iloc[0:5, :]
    #print(df_text)
          
    #frase1 = df_text.loc['sentence1']
    #frase2 = df_text.loc['sentence2']
    #frase1 = ["consumer","would","still","have","to","get","a","descramble","security","card","from","their","cable","operator","to","plug","into","the","set"]
    #frase2 = ["to","watch","pay","television","consumer","would","insert","into","the","set","a","security","card","provide","by","their","cable","service"]
    #get_vector_words_appearence(frase1, frase2)
    
    #### testing liu process
    #process_liu(frase1, frase2, ic)
    
    file_results_output = 'corpus/results.csv'
    df_results = run_dataframe_sentences(df_text, 'sentence1', 'sentence2', ic, file_results_output)
    
    #### testing liu process
    #process_agirre(frase1, frase2)
    
    print(df_results)
    


    
 
