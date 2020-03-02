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
                               , type_order = 'syntactic'):




    try:
        logger.info('reordering words in csv phrases')

        sentence = df_input[col_words_sentence].array
        

        if type_order == 'syntactic':


            #sentence = " ".join(sentence)
            sentence = list(sentence)

            print(sentence)

            #### TODO: code to up standford server API
            #### ...up syntactical parsin standford API..
            #java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
            #-preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
            #-status_port 9000 -port 9000 -timeout 15000 &

            #### ...psrsing sentence and get tree as string...
            parser = CoreNLPParser(url='http://localhost:9000')
            #p = list(parser.parse(sentence))
            p = list(parser.parse(sentence))[0]
            #### ...and we get the first syntactical tree (i'm not sure if we will have cases with several syntactical parse-trees)

            print(p)
            print('*********************')
            print('*********************')
            #p = p[0]
            str_tree = str(p)

            print(str_tree)

            #### ...get start and end of trees...
            start_tree_pattern = re.compile("\[Tree\(") #+ len('[Tree(')
            len_start_tree_pattern = len('[Tree(')
            end_tree_pattern = re.compile("\)\]")
            len_end_tree_pattern = len(')]')

            lst_pos_start_tree = [i_pos_start.start() for i_pos_start in start_tree_pattern.finditer(str_tree)]
            lst_pos_end_tree = [i_pos_end.start() + len_end_tree_pattern for i_pos_end in end_tree_pattern.finditer(str_tree)]
            lst_pos_end_tree.reverse()
            

            print(lst_pos_start_tree)
            print(lst_pos_end_tree)
            print(len(str_tree))



            ################3
            #### ...working on regex...
            
            #for 
            #####################


            #### ...we take pairs of values with start position less than end position...
            #lst_pos_start_tree = [lst_pos_start_tree[i] for i in range(0, lst_pos_start_tree) if int(lst_pos_start_tree[i]) < int(lst_pos_end_tree[i])]
            
            #print(lst_pos_start_tree)
            #print(lst_pos_end_tree)

            print('***************')

            for i in range(len(lst_pos_start_tree)):
                print(i)
                print(int(lst_pos_start_tree[i]))
                print(str_tree[ int(lst_pos_start_tree[i]) : (int(lst_pos_end_tree[i]) + len_end_tree_pattern)])
                #print(str_tree[])
                #print(str_tree[])



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


            
            print('------')
            print('------')

            for tree in parser.parse(sentence):
                
                print(tree)
                print(tree.label())
                print(tree.leaves())
                print(type(tree))
                print('-------------------')
                print(tree[0])
                print(len(tree[0]))
                
                print('-------------------')
                print(tree[0,0])
                print('-------------------')
                print(tree[0,1])

                print('-------------------')
                for subtree in tree.subtrees():
                    print(subtree)
                    print(subtree.label())
                    print(subtree.leaves())
            

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
