# -*- coding: utf-8 -*-


from similarity.wordnet_similarity_metric import *
        

####
#### FUNCTION TO GET VECTOR REPRESENTATIONS BASED ON WordNet SIMILARITY MEASURES (function get_words_similarity())
def get_vector_representation(lst_sen1, lst_sen2, type_score = 'path_similarity', corpus_ic=None):
    set_words = list(set(lst_sen1 + lst_sen2))
    #### ...we order the set of words...
    set_words = sorted(set_words)
    
    #print('###############')
    vec1 = []
    vec2 = []      
    for w in set_words:
        sim1 = []
        sim2 = []
        for w1 in lst_sen1:
            try:
                scores_sim_words = get_wordnet_words_similarity(str(w), str(w1), type_score = type_score, corpus_ic=corpus_ic)
                scores_sim_words = list(filter(lambda element: element != None, scores_sim_words))
                scores_sim_words = [0] if len(scores_sim_words) == 0 else scores_sim_words
                sim1.append(max(scores_sim_words))
            except Exception:
                sim1.append(0)
                
        for w2 in lst_sen2:
            try:
                scores_sim_words = get_wordnet_words_similarity(str(w), str(w2), type_score = type_score, corpus_ic=corpus_ic)
                scores_sim_words = list(filter(lambda element: element != None, scores_sim_words))
                scores_sim_words = [0] if len(scores_sim_words) == 0 else scores_sim_words
                sim2.append(max(scores_sim_words))
            except Exception:
                sim2.append(0)
                
        vec1.append(max(sim1))
        vec2.append(max(sim2))
                
    #print(vec1)
    #print(len(vec1)) 
    #print(vec2)
    #print(len(vec2))  

    return vec1, vec2, set_words    




if __name__=='__main__':
    lst1 = ['if', 'you', 'havent', 'you', 'should']
    lst2 = ['you', 'should', 'but', 'you', 'dont', 'have', 'to']  
    
    print(get_vector_representation(lst1, lst2, type_score = 'path_similarity', corpus_ic=None))

    

