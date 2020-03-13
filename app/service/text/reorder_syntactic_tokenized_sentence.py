# -*- coding: utf-8 -*-

from nltk.parse import CoreNLPParser
import nltk.parse.api


#### function that implements a bottom-up approach in according to a sentence syntactic tree
#### input args: tokenized sentence as list

#### TODO: code to up standford server API
#### ...up syntactical parsin standford API..
#java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
#-preload tokenize,ssplit,pos,lemma,ner,parse,depparse \
#-status_port 9000 -port 9000 -timeout 15000 &

def reorder_syntactic_tokenized_sentence(logger = None
                                        , lst_sentence = None
                                        , use_stanford_parser = True
                                        , verbose = True):

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

        #### ...we apply parsing...
        #p = list(parser.parse(sentence_reordered))
        #### ...just in case.... as string to treat with regex...
        #str_tree = str(p)
        #print(str_tree)

        #### ...we get the first syntactic root tree (just in case we have several trees)...
        lst_root_trees = []
        for tree in parser.parse(sentence_reordered):
            lst_root_trees.append(tree)
            #print(tree[0])
            #print(len(tree[0]))
            #print(tree[0,0])

        #### ...if would have a list of trees, we would pick the first...
        root_tree = lst_root_trees[0]

        #### ...we get all subtrees and load in a list...
        lst_subtrees = []
        for subtree in root_tree.subtrees():
            lst_subtrees.append(subtree)


        #### ...selecting leaves from bottom to up (aproximately)...
        #### ...trees ..are run and loaded from up to bottom... so, we reverse the list...
        lst_subtrees.reverse()
        sentence_reordered.clear()
        
        #### ...in while loop we take leaves from each tree and remove from lst_sentence (input)
        #### ...loop stop when we have runed all trees or we already take all words in original phrase
        if logger is not None:
            logger.info('syntactic reorder in input phrase: {}'.format(lst_sentence))

        i = 0
        while i < len(lst_subtrees): 
            subtree = lst_subtrees[i]
            
            leaves = subtree.leaves()
            label = subtree.label()

            for leave in leaves:
                if leave in lst_sentence_ctrl:
                    sentence_reordered = sentence_reordered + [leave] 
                    #### ...delete used word from lst_sentence_ctrl 
                    lst_sentence_ctrl.remove(leave)

            if logger is not None and verbose:
                logger.info('subtree {} from {}'.format(i+1, len(lst_subtrees)))
                logger.info(subtree)
                logger.info('tree label: {}'.format(label))
                logger.info('input phrase: {}'.format(lst_sentence))
                logger.info('tree leaves: {}'.format(leaves))
                logger.info('reordered words: {}'.format(sentence_reordered))
                logger.info('remaining words to order: {}'.format(lst_sentence_ctrl))

            i = i + 1

            #### ...end loop if lst_sentence_ctrl already is empty (all words used)
            if len(lst_sentence_ctrl) == 0:
                break

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

