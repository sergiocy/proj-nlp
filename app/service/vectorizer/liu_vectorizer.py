# -*- coding: utf-8 -*-


from nltk.corpus import wordnet as wn



def get_words_similarity(word1, word2, type_score = 'path_similarity', corpus_ic = None):    
    word_synsets_1 = wn.synsets(word1)
    word_synsets_2 = wn.synsets(word2)
    #print(word_synsets_1)
    #print(word_synsets_2)
    
    scores = []
    
    for syn1 in word_synsets_1:
        for syn2 in word_synsets_2:
            if type_score is 'path_similarity':
                scores.append(syn1.path_similarity(syn2))
            if type_score is 'res':
                scores.append(syn1.res_similarity(syn2, corpus_ic))
            if type_score is 'jcn':
                scores.append(syn1.jcn_similarity(syn2, corpus_ic))
            if type_score is 'lin':
                scores.append(syn1.lin_similarity(syn2, corpus_ic))
    #print(scores)           
    return scores
           


def get_vector_representation(lst_sen1, lst_sen2, type_score = 'path_similarity', corpus_ic=None):
    set_words = list(set(lst_sen1 + lst_sen2)) 
    
    #print('###############')
    vec1 = []
    vec2 = []      
    for w in set_words:
        sim1 = []
        sim2 = []
        for w1 in lst_sen1:
            try:
                scores_sim_words = get_words_similarity(str(w), str(w1), type_score = type_score, corpus_ic=corpus_ic)
                scores_sim_words = list(filter(lambda element: element != None, scores_sim_words))
                scores_sim_words = [0] if len(scores_sim_words) == 0 else scores_sim_words
                sim1.append(max(scores_sim_words))
            except Exception:
                sim1.append(0)
                
        for w2 in lst_sen2:
            try:
                scores_sim_words = get_words_similarity(str(w), str(w2), type_score = type_score, corpus_ic=corpus_ic)
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

    return vec1, vec2    
    


