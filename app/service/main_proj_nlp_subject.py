# -*- coding: utf-8 -*-



#from nltk.corpus import wordnet as wn
import numpy as np
import pandas as pd
from nltk import pos_tag

from input_text.input_processing import format_content_file, clean_text, get_corpus_and_ic
import vectorizer.liu_vectorizer as liu
from similarity.similarity_metric import compute_similarity_cosin, get_number_words_overlapped, get_number_synsets_overlapped, compute_pearson_coef
from graphics.histograms import graph_histogram_from_df_fields





def prepare_input(file_csv, colnames_csv, col_text1, col_text2, field_filter_name=None, field_filter_value=None, lst_gram_tags_to_del = []):

    #### GET CSV AS PANDAS DATAFRAME
    df_test = format_content_file(file_csv, colnames_csv)
    
    
    #### SUBSET OF ORIGINAL DATASET FILTERING BY A FIELD VALUE
    #### ...we can filter dataset by some field (field_filter_name) value (field_filter_value) to take a subset
    if field_filter_name is not None and field_filter_value is not None:
        df_test = df_test[df_test[field_filter_name]==field_filter_value]   
    
    #### CLEAN AND TOKENIZE TEXT 
    df_test = clean_text(df_test, col_text1, lcase=True, del_punct=True, tokenize=True, del_saxon_genitive=True, not_contraction=True)
    df_test = clean_text(df_test, col_text2, lcase=True, del_punct=True, tokenize=True, del_saxon_genitive=True, not_contraction=True)
    
    #### POSTAGGIN
    df_test[col_text1 + '_pos'] = df_test.apply(lambda x: pos_tag(x[col_text1]), axis=1) 
    df_test[col_text2 + '_pos'] = df_test.apply(lambda x: pos_tag(x[col_text2]), axis=1) 
    #### FILTERING BY POSTAGGIN
    df_test[col_text1 + '_pos_filtered'] = df_test.apply(lambda phrase_tokenized: list(filter(lambda x: x[1] not in lst_gram_tags_to_del, phrase_tokenized[col_text1 + '_pos'])), axis=1) 
    df_test[col_text2 + '_pos_filtered'] = df_test.apply(lambda phrase_tokenized: list(filter(lambda x: x[1] not in lst_gram_tags_to_del, phrase_tokenized[col_text2 + '_pos'])), axis=1) 
    df_test[col_text1 + '_filtered'] = df_test.apply(lambda phrase_tagged: [w[0] for w in phrase_tagged[col_text1 + '_pos_filtered']], axis=1) 
    df_test[col_text2 + '_filtered'] = df_test.apply(lambda phrase_tagged: [w[0] for w in phrase_tagged[col_text2 + '_pos_filtered']], axis=1) 
    
    #### SENTENCES CLEANED AND FILTERED AS STRING
    #### ...corpus getted from filtered (by grammar label) phrases
    df_test[col_text1 + '_str'] = df_test.apply(lambda x: ' '.join(x[col_text1 + '_filtered']), axis=1) 
    df_test[col_text2 + '_str'] = df_test.apply(lambda x: ' '.join(x[col_text2 + '_filtered']), axis=1) 
    
    return df_test
    
    


def process_liu(sen1, sen2, ic, type_similarity = 'res'):
    print('LAUNCHING LIU PROCESS WITH SIMILARITY {0}'.format(type_similarity))
    print(sen1)
    vector1, vector2 = liu.get_vector_representation(sen1, sen2, type_score=type_similarity, corpus_ic=ic)

    #### restriction for 'jcn' similarity...
    #### ...we consider only 1 (vector element = 1e+300) or 0 values (vector element < 1)
    if type_similarity == 'jcn':
        vector1 = np.asarray(vector1)
        vector2 = np.asarray(vector2)
        #vector1[np.where(vector1 == 1e+300)] = 1
        #vector1[np.where(vector1 < 1)] = 0
        vector1 = np.where(vector1 < 100, int(0), vector1)
        vector1 = np.where(vector1 > 1000000, int(1), vector1)
        vector2 = np.where(vector2 < 100, int(0), vector2)
        vector2 = np.where(vector2 > 1000000, int(1), vector2)
    
    magn, cos, ang = compute_similarity_cosin(list(vector1), list(vector2))
    ang = (360*ang)/(2*3.1416)
            
    return list(vector1), list(vector2), magn, cos, ang




def process_agirre(sen1, sen2):
    print('LAUNCHING AGIRRE PROCESS')
    
    n_overlapped = get_number_words_overlapped(sen1, sen2)
    agirre_metric = (2*n_overlapped)/(len(sen1)+len(sen2))
        
    lst_syn_overlapped, n_syn_overlapped = get_number_synsets_overlapped(sen1, sen2)
    agirre_metric_synsets = (2*n_syn_overlapped)/(len(sen1)+len(sen2))
    
    return agirre_metric, agirre_metric_synsets
    



def run_compute_similarities(df, field_text1, field_text2, ic, file_output_dataframe):
    print('RUNNING PROCESS...')
    
    df['vector1'] = df.apply(lambda x: process_liu(list(x[field_text1]), list(x[field_text2]), ic, type_similarity = 'res')[0], axis=1)
    df['vector2'] = df.apply(lambda x: process_liu(list(x[field_text1]), list(x[field_text2]), ic, type_similarity = 'res')[1], axis=1)
    
    df['liu_path'] = df.apply(lambda x: process_liu(list(x[field_text1]), list(x[field_text2]), ic, type_similarity = 'path_similarity')[3], axis=1) 
    df['liu_res'] = df.apply(lambda x: process_liu(list(x[field_text1]), list(x[field_text2]), ic, type_similarity = 'res')[3], axis=1) 
    df['liu_lin'] = df.apply(lambda x: process_liu(list(x[field_text1]), list(x[field_text2]), ic, type_similarity = 'lin')[3], axis=1)
    df['liu_jcn'] = df.apply(lambda x: process_liu(list(x[field_text1]), list(x[field_text2]), ic, type_similarity = 'jcn')[3], axis=1) 
    
    df['agirre_words'] = df.apply(lambda x: process_agirre(list(x[field_text1]), list(x[field_text2]))[0], axis=1) 
    df['agirre_synsets'] = df.apply(lambda x: process_agirre(list(x[field_text1]), list(x[field_text2]))[1], axis=1)
    
    df.to_csv(file_output_dataframe, sep=';')
    
    return df




def run_evaluate_results(df, file_output_dataframe):
    
    #### ...to compute similarities, we will remove registers with NULL similarity...
    #### That is happened because sometimes we have vectors=[0] 
    df = df[df['score'].notnull()]
    #n_total_registers = df_aux.shape[0]
    #print(df_aux)
    #print(df_aux.shape[0])
    
    df_aux = df[df['liu_path'].notnull()]
    var1 = df_aux['score']
    var2 = df_aux['liu_path']        
    df_scores = pd.DataFrame([['liu_path', list(compute_pearson_coef(var1, var2)[0])[0], list(compute_pearson_coef(var1, var2)[0])[1], df_aux.shape[0]]], columns = ['type_score', 'pearson', 'p-value', 'n_registers']) 
    
    df_aux = df[df['liu_res'].notnull()]
    var1 = df_aux['score']
    var2 = df_aux['liu_res']
    df_scores = df_scores.append({'type_score':'liu_res', 'pearson':list(compute_pearson_coef(var1, var2)[0])[0], 'p-value':list(compute_pearson_coef(var1, var2)[0])[1], 'n_registers':df_aux.shape[0]}, ignore_index=True)
    
    df_aux = df[df['liu_lin'].notnull()]
    var1 = df_aux['score']
    var2 = df_aux['liu_lin']
    df_scores = df_scores.append({'type_score':'liu_lin', 'pearson':list(compute_pearson_coef(var1, var2)[0])[0], 'p-value':list(compute_pearson_coef(var1, var2)[0])[1], 'n_registers':df_aux.shape[0]}, ignore_index=True)
    
    df_aux = df[df['liu_jcn'].notnull()]
    var1 = df_aux['score']
    var2 = df_aux['liu_jcn']
    df_scores = df_scores.append({'type_score':'liu_jcn', 'pearson':list(compute_pearson_coef(var1, var2)[0])[0], 'p-value':list(compute_pearson_coef(var1, var2)[0])[1], 'n_registers':df_aux.shape[0]}, ignore_index=True)
    
    df_aux = df[df['agirre_words'].notnull()]
    var1 = df_aux['score']
    var2 = df_aux['agirre_words']
    df_scores = df_scores.append({'type_score':'agirre_words', 'pearson':list(compute_pearson_coef(var1, var2)[0])[0], 'p-value':list(compute_pearson_coef(var1, var2)[0])[1], 'n_registers':df_aux.shape[0]}, ignore_index=True)
    
    df_aux = df[df['agirre_synsets'].notnull()]
    var1 = df_aux['score']
    var2 = df_aux['agirre_synsets']
    df_scores = df_scores.append({'type_score':'agirre_synsets', 'pearson':list(compute_pearson_coef(var1, var2)[0])[0], 'p-value':list(compute_pearson_coef(var1, var2)[0])[1], 'n_registers':df_aux.shape[0]}, ignore_index=True)
    
    
    df_scores.to_csv(file_output_dataframe, sep=';')
    
    
    return df_scores
    
               
       

    

if __name__ == '__main__':
    
    ###################################################
    #### EXECUTION PARAMETERS
    print('***** PARAMETERS *****')
    #file_dev = 'corpus/sts-dev.csv'
    #file_train = 'corpus/sts-train.csv'
    file_test = 'corpus/stsdatasets/sts-test.csv' 
    colnames_csv = ['genre', 'filename', 'year', 'number', 'score', 'sentence1', 'sentence2']
    colname_csv_text1 = 'sentence1'
    colname_csv_text2 = 'sentence2'
    
    colname_csv_field_subsetting = None #field_filter_name='genre'
    colname_csv_value_subsetting = None #field_filter_value='main-captions'
    lst_type_grammatical_words_to_del = ['DT', 'IN', 'CC'] # ['DT', 'IN'] ... ['CC'] conjunciones
    
    colname_csv_text1_processed = 'sentence1_str'
    colname_csv_text2_processed = 'sentence2_str'
    colname_csv_text1_processed_tokenized = 'sentence1_filtered'
    colname_csv_text2_processed_tokenized = 'sentence2_filtered'
    
    file_corpus_output = 'corpus/corpus_to_compute_ic.txt'
    file_similarities_output = 'corpus/similarities.csv'
    file_results_output = 'corpus/results.csv'


    ###################################################
    #### LOAD AND CLEANING DATA 
    print('***** loading data *****')
    df_text = prepare_input(file_test, colnames_csv, colname_csv_text1, colname_csv_text2, field_filter_name=colname_csv_field_subsetting, field_filter_value=colname_csv_value_subsetting, lst_gram_tags_to_del = lst_type_grammatical_words_to_del)
    #### ...we get the corpus used to compute IC in a file using phrases filtered...
    lst_sentences = list(np.append(df_text[colname_csv_text1_processed], df_text[colname_csv_text2_processed]))
    corpus_for_ic, ic = get_corpus_and_ic(set_of_strings=lst_sentences, file_corpus_output=file_corpus_output)
    print('***********************************')
    
    
    ###################################################
    #### COMPUTE SIMILARITIES AND GENERATE A RESULTS FILE
    print('***** computing similarities *****')
    #### ...we can select a few rows to develope...
    #df_text = df_text.loc[0:10]
    #df_text = df_text.iloc[73:77,]
    
    #### ...we compute similarities for each pair of phrases in csv...
    df_results = run_compute_similarities(df_text, colname_csv_text1_processed_tokenized, colname_csv_text2_processed_tokenized, ic, file_similarities_output)    
    
    #### ...and we can plot an histogram  with elements in all vectors, to see how that components are distributed
    #graph_histogram_from_df_fields(df_results, ['vector1', 'vector2'])
    print('***********************************')
    
    ###################################################
    #### COMPARE RESULTS
    print('***** studying results *****')
    df_results = pd.read_csv(file_similarities_output, sep=';') 
    #print(df_results)
    
    #### ...to develope, we try a subset...
    #df_results = df_results.loc[0:3,]
    
    df_scores = run_evaluate_results(df_results, file_results_output)
    print(df_scores)
    print('***********************************')
    


    
 
