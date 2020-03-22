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
                                            , logger = None
                                            , verbose = True
                                            , start_nest = '['
                                            , end_nest = ']'):

        if logger is not None:
            logger.info(' ******** input nested structure : {}'.format(str_structure))

        lst_structure = list(str_structure)

        ####
        ######## STEP 1: detect positions of nest start-symbols and end-symbols
        #### ...we get the positions of start nest and end nest as boolean lists...
        lst_bool_start_nest_pos = [True if lst_structure[i_chr] == start_nest else False for i_chr in range(len(lst_structure))]
        lst_bool_end_nest_pos = [True if lst_structure[i_chr] == end_nest else False for i_chr in range(len(lst_structure))]

        df_nested_structure = pd.DataFrame({'symbol_start': []
                                            , 'symbol_end': []
                                            , 'layer_x': []
                                            , 'layer_depth': []
                                            , 'father_layer_x': []
                                            , 'father_layer_depth':[]
                                            , 'pos_char_start': []
                                            , 'pos_char_end': []
                                            , 'string_fragment': []})

        ####
        ######## STEP 2: detect nested substructures
        #lst_ctrl = lst_structure
        #ctrl = True
        while True:
            #### ...initial positions of root-tree...
            current_layer_x = 0
            current_layer_depth = -1
            #### ...initial positions on lists of detected parenthesis...
            current_start_pos = 0
            current_end_pos = 0

            #### ...run on start parenthesis matched...
            for i_start in range(len(lst_bool_start_nest_pos)):
                #print('*** character {0} - position {1}'.format(lst_structure[i_start], i_start))

                if lst_bool_start_nest_pos[i_start]:
                    #print('*** start parenthesis found: {}'.format(lst_structure[i_start]))
                    current_start_pos = i_start
                    current_layer_depth = current_layer_depth + 1

                    for i_end in range(i_start+1, len(lst_bool_start_nest_pos)):
                        #print('*** searching end parenthesis found: {}'.format(lst_structure[i_end]))
                        if lst_bool_start_nest_pos[i_end]:
                            break
                        if lst_bool_end_nest_pos[i_end]:
                            current_end_pos = i_end
                            break

                    if current_end_pos > 0:
                        if logger is not None and verbose:
                            logger.info('**** nested parenthesis found!! ****')
                            logger.info('start pos: {0} - end pos: {1} - expresion: {2}'.format(current_start_pos, current_end_pos, ''.join(lst_structure[current_start_pos:current_end_pos+1])))
                            logger.info('layer_x: {0} - layer_depth: {1}'.format(current_layer_x, current_layer_depth))

                        #### ...define current_layer_x...
                        if len(df_nested_structure[df_nested_structure['layer_depth'] == current_layer_depth]) > 0:
                            df_nested_x = df_nested_structure[df_nested_structure['layer_depth'] == current_layer_depth]
                            #df_nested_x = df_nested_x.sort_values(by=['pos_char_start'])
                            current_layer_x = df_nested_x['layer_x'].max() + 1

                        #### ...add substructure found in output dataframe...
                        lst_row_to_append = [start_nest, end_nest, current_layer_x, current_layer_depth, 0, 0, current_start_pos, current_end_pos, ''.join(lst_structure[current_start_pos:current_end_pos+1])]
                        row_to_append = pd.Series(lst_row_to_append, index = df_nested_structure.columns)
                        df_nested_structure = df_nested_structure.append(row_to_append, ignore_index=True)

                        #### ...remove parenthesis found...
                        lst_bool_start_nest_pos[current_start_pos] = False
                        lst_bool_end_nest_pos[current_end_pos] = False
                        break

            #### ...if all parenthesis found ctrl = False and finish while loop...
            if True not in lst_bool_start_nest_pos or True not in lst_bool_end_nest_pos:
                if True not in lst_bool_start_nest_pos and True not in lst_bool_end_nest_pos:
                    #ctrl = False
                    break
                else:
                    if logger is not None:
                        logger.error(' - troubles with nested string: {0}'.format(str_structure))
                    raise Exception



        ####
        ######## STEP 3: set father node/tree (setting its indexes layer_x and layer_depth)

        #### ...we define the substructures length; with this we can get which structure contains which...
        df_nested_structure['substring_length'] = df_nested_structure['pos_char_end'] - df_nested_structure['pos_char_start']

        #### ...we get, for each nested-strucuture found, the structure shorter containining it...
        for i_nest, row in df_nested_structure.iterrows():
            pos_start = df_nested_structure.iloc[i_nest]['pos_char_start']
            pos_end = df_nested_structure.iloc[i_nest]['pos_char_end']

            df_nest_aux = df_nested_structure[(df_nested_structure['pos_char_start'] < pos_start) & (df_nested_structure['pos_char_end'] > pos_end)]
            df_nest_aux = df_nest_aux[df_nest_aux['substring_length'] == df_nest_aux['substring_length'].min()]

            if len(df_nest_aux) > 0:
                df_nested_structure.loc[i_nest, 'father_layer_x'] = df_nest_aux.iloc[0]['layer_x']
                df_nested_structure.loc[i_nest, 'father_layer_depth'] = df_nest_aux.iloc[0]['layer_depth']


        df_nested_structure = df_nested_structure.drop('substring_length', axis = 1)

        return df_nested_structure




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
        df_tree_structure = parse_nested_structure_with_regex(str_tree, logger = logger, verbose = True)

        print(df_tree_structure)




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
