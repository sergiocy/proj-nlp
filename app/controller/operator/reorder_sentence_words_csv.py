# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
#import nltk
from nltk.parse import CoreNLPParser
import nltk.parse.api
#import nltk





#### READING dataframe with phrases in one of its columns nd reorder to apply vector composition in this order
def reorder_sentence_words_csv(logger = None
                               , df_input = None
                               , col_words_sentence = None
                               , type_order = 'syntactic'
                               , use_stanford_parser = True
                               , verbose = True):


    #### input args: tokenized sentence as list
    def reorder_tokenized_sentence(logger = None
                                   , lst_sentence = None
                                   , use_stanford_parser = True
                                   , verbose = True):

        sentence_reordered = lst_sentence

        if use_stanford_parser:
            parser = CoreNLPParser(url='http://localhost:9000')

        if logger is not None:
            



        return sentence_reordered




    try:
        logger.info('reordering words in csv phrases')

        sentence = df_input[col_words_sentence].array
        

        if type_order == 'syntactic':


            #sentence = " ".join(sentence)
            sentence = list(sentence)
            logger.info('input phrase: {}'.format(sentence))

            #### TODO: code to up standford server API
            #### ...up syntactical parsin standford API..
            #java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
            #-preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
            #-status_port 9000 -port 9000 -timeout 15000 &

            #### ...psrsing sentence and get tree as string...
            if use_stanford_parser:
                parser = CoreNLPParser(url='http://localhost:9000')
            #p = list(parser.parse(sentence))
            p = list(parser.parse(sentence))
            str_tree = str(p)
            print(p)
            print(str_tree)
            #### ...and we get the first syntactical tree (i'm not sure if we will have cases with several syntactical parse-trees)

            
            print('*********************')
            print('*********************')
            
            #### 
            #### ...approach based on regular expressions...
            #### ...get start and end of trees...
            #start_tree_pattern = re.compile("\[Tree\(") #+ len('[Tree(')
            #len_start_tree_pattern = len('[Tree(')
            #end_tree_pattern = re.compile("\)\]")
            #len_end_tree_pattern = len(')]')
            #lst_pos_start_tree = [i_pos_start.start() for i_pos_start in start_tree_pattern.finditer(str_tree)]
            #lst_pos_end_tree = [i_pos_end.start() + len_end_tree_pattern for i_pos_end in end_tree_pattern.finditer(str_tree)]
            #lst_pos_end_tree.reverse()
            #print(lst_pos_start_tree)
            #print(lst_pos_end_tree)
            #print(len(str_tree))

            
            #for i in range(len(lst_pos_start_tree)):
            #    print(i)
            #    print(int(lst_pos_start_tree[i]))
            #    print(str_tree[ int(lst_pos_start_tree[i]) : (int(lst_pos_end_tree[i]) + len_end_tree_pattern)])
            #    #print(str_tree[])
            #    #print(str_tree[])





            '''
            if len(lst_pos_start_tree) != len(lst_pos_start_tree):
                print("start and end of brackets inconsistent")
                raise Exception

            else:

                for pos_start_tree in lst_pos_start_tree:
                    str_subtree = str_tree[pos_start_tree]

            #str_sub_tree = str_tree[i_start_tree_pattern:i_end_tree_pattern]

            #print(str_sub_tree)

            #for m in p.finditer(str_tree):
            #    print(m.start(), m.group())
            #    print(str_tree[m.start():len(str_tree)])

            #start = str_tree.find('\[Tree\(')
            #end = s.find('ZZZ', start)
            #s[start:end]
            #print(str_tree[start:len(str_tree)])

            
            for i in p:
                print(i)
                print(type(i))
            '''


            print('***************')
            print('------')
            print('------')

            ####
            #### ...we get the first syntactic root tree (just in case we have several trees)...
            lst_root_trees = []
            for tree in parser.parse(sentence):
                lst_root_trees.append(tree)
                #print(tree[0])
                #print(len(tree[0]))
                #print(tree[0,0])
            
            root_tree = lst_root_trees[0]

            print(root_tree)
            print(root_tree.label())
            print(root_tree.leaves())

            ####
            #### ...we get all subtrees
            lst_subtrees = []
            for subtree in root_tree.subtrees():
                lst_subtrees.append(subtree)

            print('***************')
            print('------')
            print('------')

            for t in lst_subtrees:
                print('------')
                print(t)


            

        elif type_order == 'direct':
            sentence = sentence
            print(sentence)

        elif type_order == 'reverse':
            sentence = list(sentence)
            sentence.reverse()

            sentence = np.asarray(sentence)
            print(sentence)

        else:
            logger.info('this type_order not exist')
            raise Exception


    except Exception:
        if logger is not None:
            logger.exception("ERROR reading csv")

        raise Exception

    #return df
