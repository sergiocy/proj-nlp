#!/usr/bin/env python
# -*- coding: utf-8 -*-


#import sys
import numpy as np
import nltk
from nltk import load_parser
#from nltk import sem




def get_token_velocity(tkns):
    
    preps = ['por' , 'cada']
    magnitudes = ['centimetro'
                  , 'centimetros'
                  , 'kilometro'
                  , 'kilometros'
                  , 'segundo'
                  , 'segundos'
                  , 'metro'
                  , 'metros'
                  , 'hora'
                  , 'horas']
    
    tkn = 0 
    while tkn < len(tkns):    
        if tkns[tkn] in preps:
            if tkns[tkn-1] in magnitudes and tkns[tkn+1] in magnitudes:
                tkns[tkn] = tkns[tkn-1] + " " + tkns[tkn] + " " + tkns[tkn+1]            
                tkns.pop(tkn+1)
                tkns.pop(tkn-1)
        
        tkn = tkn + 1
    
    return tkns



####
#### function to get a summary of semantic from a parsed tree
def get_main_node_sem(tree):
    
    s_node = tree.label()    
    s_node = s_node['SEM']    
    
    semantic_features = [s_node.get('ACCION'), s_node.get('MAGNITUD'), s_node.get('COMPLEMENTO'), s_node.get('CARGA')]
    
    iter_features = 0
    for node in semantic_features:

        if isinstance (node, nltk.sem.logic.Variable):
            semantic_features[iter_features] = ''            
        if type(node) is nltk.grammar.FeatStructNonterminal:
            semantic_features[iter_features] = []
            valor = node.get('VALOR')
            unidad = node.get('UNIDAD')
            if isinstance (valor, nltk.sem.logic.Variable):
                valor = ''
            if isinstance (unidad, nltk.sem.logic.Variable):
                unidad = ''
            semantic_features[iter_features] = [valor, unidad]

        iter_features = iter_features + 1
    
    return tree, s_node, semantic_features



####
#### function to detect semantic counting the maximum features with value
def get_semantic(trees, features, semantic):
    
    counting = []
    for sem in semantic:
        count = 0
        for feature_father in sem:
            if type(feature_father) is str and feature_father is not '':
                count = count + 1
            if type(feature_father) is list:
                for feature_child in feature_father:
                    if type(feature_child) is str and feature_child is not '': 
                        count = count + 1
        
        counting.append(count) 
    
               
    max_str = np.amax( np.asarray(counting) )
    index_max_str = np.asarray( np.where(counting == max_str) )[0]
    
    filtered_trees = []
    filtered_features = []
    filtered_semantic = []
    filtered_trees = [trees[one_sem] for one_sem in index_max_str]
    filtered_features = [features[one_sem] for one_sem in index_max_str]
    filtered_semantic = [semantic[one_sem] for one_sem in index_max_str]
        
    return filtered_trees, filtered_features, filtered_semantic
    
    



if __name__ == '__main__':
    
    grammar_file = 'gramatica_base.fcfg' 
    #text_file = 'textos_dev.txt'
    text_file = 'textos_test.txt'
    
    #cp = load_parser(grammar_file, trace=2) #### trace=2 to show all steps in parsing
    cp = load_parser(grammar_file)

    infile = open(text_file, encoding = 'utf8')
    outfile = open('output-test.txt','w')
    
    # We analyze each line in input file 
    for line in infile:
        print('\n *************** NEW PHRASE ***************')
        print(line)
        tokens=line.split()
        
        #print(tokens)
        tokens = get_token_velocity(tokens)
        #print(tokens)
        
        trees = cp.parse(tokens)
        
        trees_parsed = []
        features_structure_parsed = []
        semantic_parsed = []
        
        for tree in trees: 
            tree, s_node_features, semantic_features  = get_main_node_sem(tree)
            #print(tree)
            #print(s_node_features)
            #print(semantic_features)
            trees_parsed.append(tree)
            features_structure_parsed.append(s_node_features)
            semantic_parsed.append(semantic_features)

        ####
        #### we select parsing with all semantic features detected
        trees_parsed, features_structure_parsed, semantic_parsed = get_semantic(trees_parsed, features_structure_parsed, semantic_parsed)
        
        
        for i in range(len(semantic_parsed)):
            if (semantic_parsed[i][0] == 'indicar' or 'quedar') and (semantic_parsed[i][3] == 'bateria'):
                semantic_parsed[i] = ['estado_bateria', '', '', '']
            #print(trees_parsed[i])
            #print(features_structure_parsed[i])
            print(semantic_parsed[i])
            
            outfile.write('\n **** \n' + line + str(semantic_parsed[i]) + '\n')

    
    outfile.close()
    infile.close()
