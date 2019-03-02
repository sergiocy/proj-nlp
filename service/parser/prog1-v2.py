hon# -*- coding: utf-8 -*-


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist



####
#### FUNCTION TO DEFINE THE DICTIONARY
def load_dict():

    dict ={}
    dict['.']='PUNT'
    dict[',']='PUNT'
    dict['(']='PUNT'
    dict[')']='PUNT'
    dict[';']='PUNT'

    dict['la']='ART'
    dict['el']='ART'
    dict['los']='ART'
    dict['las']='ART'
    dict['del']='ART'
    dict['al']='ART'
    
    dict['un']='DET'
    dict['una']='DET'
    dict['su']='DET'
    dict['sus']='DET'
    dict['cualquier']='DET'
    
    dict['a']='PREP'
    dict['para']='PREP'
    dict['en']='PREP'
    dict['de']='PREP'
    dict['por']='PREP'
    dict['como']='PREP'
    dict['con']='PREP'
    dict['desde']='PREP'
    
    dict['que']='CONJ'
    dict['y']='CONJ'
    dict['o']='CONJ'
    dict['u']='CONJ'
    
    dict['se']='PRON'
    dict['lo']='PRON'
    dict['otro']='PRON'
    
    dict['más']='ADV'
    dict['así']='ADV'
    
    dict['personal']='ADJ'
    dict['físico']='ADJ'
    dict['convenientes']='ADJ'
    dict['útiles']='ADJ'
    dict['operativo']='ADJ'
    dict['lógico']='ADJ'
    dict['digital']='ADJ'
    dict['complejo']='ADJ'

    
    #### SUSTANTIVOS
    dict['sistema']='NCMS'
    dict['sistemas']='NCMP'
    dict['programa']='NCMS'
    dict['programas']='NCMP'
    dict['código']='NCMS'
    dict['códigos']='NCMP'
    dict['circuito']='NCMS'
    dict['circuitos']='NCMP'
    dict['unidad']='NCFS'
    dict['unidades']='NCFP'
    
    dict['informática']='NCFN'
    dict['computadora']='NCFS'
    dict['instrucciones']='NCFP'
    dict['dispositivo']='NCMS'
    dict['máquina']='NCFS'
    dict['central']='NCFS'
    dict['datos']='NCMP'

    dict['conjunto']='NCMS'
    dict['tipo']='NCMS'
    dict['componentes']='NCMP'
    dict['cpu']='NCMS'
    dict['microprocesador']='NCMS'
    dict['microcontrolador']='NCMS'
    dict['operaciones']='NCFP'
    dict['partes']='NCFP'
    dict['ordenador']='NCMS'
    dict['realización']='NCFS'
    dict['interacción']='NCFS'
    
    dict['procesamiento']='NCMS'
    dict['formato']='NCMS'
    dict['fuente']='NCFS'

    dict['salida']='NCFS'
    dict['autómata']='NCMS'
    dict['lenguaje']='NCMS'
    
    dict['estructura']='NCFS'
    dict['pequeñas']='NCFP'
    
    #### VERBOS
    # SER
    dict['es']='VMIP3S0'
    # INCLUIR
    dict['incluye']='VMIP3S0'
    # PERMITIR
    dict['permite']='VMIP3S0'
    # CONOCER
    dict['conoce']='VMIP3S0'
    # TENER
    dict['tiene']='VMIP3S0'
    # LEER
    dict['lee']='VMIP3S0'
    # REALIZAR
    dict['realiza']='VMIP3S0'
    # CONVERTIR
    #dict['convertirlos']
    # ENVIAR
    dict['envían']='VMIP3P0'
    
    return dict


####
#### FUNCTION TO DEFINE REGULAR EXPRESSSIONS USED
def load_regex():
    #Aquí hay se añaden los patrones necesarios
    p=[
    # ...anglicismos "hardware", "software"...
    (r'.*ware$', 'NCMS'),
    # ...palabras con terminaciones "-ble"
    (r'.*ble(|s)$', 'ADJ'),
    (r'.*mente(|s)$', 'ADV'),
    (r'.*ic(o|a)(|s)$', 'ADJ'),
    # ...los infinitivos gerundios y participios
    (r'.*(ar|er|ir)$', 'VMN0000'),
    (r'.*ada$', 'VMP000F'),
    (r'.*ado$', 'VMP000M'),
    (r'.*(a|e)ndo$', 'VMG0000'),
    # ...otras palabras...
    (r'.*$','OTHER')
    ]
    
    return p


####
#### FUNCTION TO SHOW ALL WORDS AND ITS TAGS
def show_words_tagged(taggedText):
    for item in taggedText:
        if item[0] in dict:
            print(item[0]+' '+dict[item[0]])
        else:
            print(item[0]+'  '+item[1])

####
#### FUNCTION TO SHOW THE WORDS TAGGED WITH ONE SPECIFIC TAG (type_word) 
def show_type_of_words_tagged(taggedText, type_word):
    for item in taggedText:
        if item[1] == type_word:
            print(item[0] + '  ' + item[1])
            
####
#### FUNCTION TO SHOW A WORD AND ITS TAG            
def show_word_tagged(taggedText, word):
    for item in taggedText:
        if item[0] == word:
            print('REGEX TAGGING: ' + item[0] + '  ' + item[1])
            if word in dict:
                print('DICT TAGGING: ' + item[0] + '  ' + dict[item[0]])

####
#### FUNCTION TO SAVE WORDS AND ITS TAGS IN A DICTIONARY ("results" variable)
def save_words_tagged(taggedText):
    results = []
    
    for item in taggedText:
        if item[0] in dict:
            results.append([item[0], dict[item[0]]])
        else:
            results.append([item[0], item[1]])
            
    return results

####
#### FUNCTION TO SAVE WORDS AND ITS TAGS IN A DICTIONARY ("results" variable)
def write_words_tagged(results):
    file_output = open('output-test.txt','w') 
    
    for w in results:
        file_output.write(w[0] + ' - ' + w[1] + '\n')
    
    file_output.close()
    
    
    
if __name__ == '__main__':
    f = open ('texto_test.txt', encoding = 'utf8')
    
    words=nltk.word_tokenize(f.read())
    words=[word.lower() for word in words]
    #print(len(words)) #total words: 662
    
    #fd = nltk.FreqDist(word.lower() for word in words)
    #fdf= fd.most_common(100)

    dict = load_dict()
    p = load_regex()
    rt=nltk.RegexpTagger(p)
    taggedText=rt.tag(words)
    
    results = save_words_tagged(taggedText)
    #print(save_words_tagged(taggedText))
    write_words_tagged(results)
    
        