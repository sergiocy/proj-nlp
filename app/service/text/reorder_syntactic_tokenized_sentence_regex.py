# -*- coding: utf-8 -*-

import pandas as pd
from nltk.parse import CoreNLPParser
import nltk.parse.api


#### function that implements a bottom-up approach in according to a sentence syntactic tree
#### input args: tokenized sentence as list

#### TODO: code to up standford server API
#### ...up syntactical parsin standford API..
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
#-preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
#-status_port 9000 -port 9000 -timeout 15000 &

def reorder_syntactic_tokenized_sentence_regex(logger = None
                                        , lst_sentence = None
                                        , use_stanford_parser = True
                                        , verbose = True):




    #### WARNING!!! needed that str_structure betwwen start_nest and end_nest to get a root-point in tree (as nested sting)
    def parse_nested_structure_with_regex(str_structure
                                            , start_nest = '['
                                            , end_nest = ']'):

        print('\n #############################')
        print('nested structure : {}'.format(str_structure))

        lst_structure = list(str_structure)
        #print(lst_structure)

        #patron = re.compile(r'\bfoo\b')
        #lst_start_nest_pos = start_nest.search(str_structure)
        #print(lst_start_nest_pos)

        #### ...we get the positions of start nest and end nest as boolean lists...
        lst_bool_start_nest_pos = [True if lst_structure[i_chr] == start_nest else False for i_chr in range(len(lst_structure))]
        lst_bool_end_nest_pos = [True if lst_structure[i_chr] == end_nest else False for i_chr in range(len(lst_structure))]

        #print(lst_bool_start_nest_pos)
        #print(lst_bool_end_nest_pos)


        print('****************')
        df_nested_structure = pd.DataFrame({'layer_x': []
                                            , 'layer_depth': []
                                            , 'pos_char_start': []
                                            , 'pos_char_end': []})
        print(df_nested_structure)


        #lst_ctrl = lst_structure
        #while

        #### ...initial positions of root-tree...
        current_layer_x = 0
        current_layer_depth = 0
        #### ...initial positions on lists of detected parenthesis...
        current_start_pos = 0
        current_end_pos = 0

        #### ...run on start parenthesis matched...
        for i_start in range(len(lst_bool_start_nest_pos)):
            print('*** character {}'.format(lst_structure[i_start]))

            if lst_bool_start_nest_pos[i_start]:
                print('*** start parenthesis found: {}'.format(lst_structure[i_start]))
                current_start_pos = i_start
                current_layer_depth = current_layer_depth + 1

                for i_end in range(i_start+1, len(lst_bool_start_nest_pos)):
                    print('*** searching end parenthesis found: {}'.format(lst_structure[i_end]))
                    if lst_bool_start_nest_pos[i_end]:
                        break
                    if lst_bool_end_nest_pos[i_end]:
                        current_end_pos = i_end
                        break

                if current_end_pos > 0:
                    print('**** nested parenthesis found!! ****')
                    print('start pos: {0} - end pos: {1} - expresion: {2}'.format(current_start_pos, current_end_pos, ''.join(lst_structure[current_start_pos:current_end_pos])))
                    break


        print('\n #############################')

        #return str_structure, df_nested_structure



    try:
        #### ...to control words that we used...
        lst_sentence_ctrl = list(lst_sentence)
        #### ...to store output...
        sentence_reordered = list(lst_sentence)

        parser = None
        if use_stanford_parser:
            parser = CoreNLPParser(url='http://localhost:9000')
        else:
            if logger is not None:
                logger.warn('we need you up standford nlp parser')



        ####
        ####
        lst_root_trees = []
        for tree in parser.parse(sentence_reordered):
            lst_root_trees.append(tree)
        #### ...if would have a list of trees, we would pick the first...
        root_tree = lst_root_trees[0]
        print(root_tree)



        #### ...we apply parsing...
        p = list(parser.parse(sentence_reordered))
        #### ...just in case.... as string to treat with regex...
        str_tree = str(p)
        parse_nested_structure_with_regex(str_tree)




        if logger is not None:
            logger.info('reordered sentence: '.format(sentence_reordered))

        #### ...warning if for some trouble, len(input sentence) != len(output sentence)
        if logger is not None and len(lst_sentence) != len(sentence_reordered):
            logger.warn('input and output sentences with different length')


    except Exception as e:
        if logger is not None:
            logger.exception('ERROR reordering phrase: {}'.format(lst_sentence))
        raise e


    return sentence_reordered
