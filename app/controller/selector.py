
import nltk
from service.inputtext.InputText import InputText


def process_selection(param_config):
    print(param_config.get('process') + ' - ' + param_config.get('file_origin'))
    file_origin = param_config.get('file_folder') + param_config.get('file_origin')
    print(param_config.get('process') + ' - ' + param_config.get('file_origin'))
    
    if param_config.get('process') == 'wordsemb':
        txt_input = set_input_object(param_config, file_origin)
        print(txt_input)
        #call_wordsemb(file_origin)



def set_input_object(param_config, file_origin):
    print('aqui hay que setear el objeto de entrada')

    in_text1 = InputText(file_origin)
    in_text1.set_output_text(lcase = param_config.get('txt_lcase')
                             , tokenize = param_config.get('txt_tokenize')
                             )

    return in_text1.get_output_text()



def call_wordsemb(file_origin):
    try:
        f = open (file_origin[0], encoding = 'utf8')
        txt = f.read()
        print(txt)

        words=nltk.word_tokenize(txt)
        words=[word.lower() for word in words]
        print(words)
        print(len(words))

    except Exception as e:
        print(e)
    




if __name__ == "__main__":
    print('using selector.py')    