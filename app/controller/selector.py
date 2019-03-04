
import nltk
from service.inputtext.InputText import InputText


def process_selection(file_path, process, file_origin):
    print(process[0] + ' - ' + file_origin[0])
    file_origin = file_path + '/' + file_origin[0]
    print(process[0] + ' - ' + file_origin)
    
    if process[0] == 'wordsemb':
        set_input_object(file_origin)
        #call_wordsemb(file_origin)



def set_input_object(file_origin):
    print('aqui hay que setear el objeto de entrada')

    print('creating class for input-text')
    in_text1 = InputText(file_origin)
    in_text1.get_file_name()
    in_text1.get_input_text()
    in_text1.set_output_text(lcase = True, tokenize = True)
    in_text1.get_output_text()



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