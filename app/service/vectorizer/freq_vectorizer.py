# -*- coding: utf-8 -*-


from nltk.corpus import wordnet as wn


# -*- coding: utf-8 -*-  


def get_vector_words_appearence(sen1, sen2):
    set_words = list(set(sen1 + sen2)) 
    print(set_words)
    print(sen1)
    print(sen2)
    
    #vec1 = list(pd.Series(sen1).apply(lambda w: 1 if w in set_words else 0))
    #vec2 = list(pd.Series(sen2).apply(lambda w: 1 if w in set_words else 0))

    vec1 = [1 if w in sen1 else 0 for w in set_words]
    vec2 = [1 if w in sen2 else 0 for w in set_words]

    #sen1 = np.where(np.asarray(sen1) in set_words, 1, 0)    
    #print(sen1)
    print(vec1)
    print(vec2)

