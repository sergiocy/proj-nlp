
import nltk


def process_selection(process, file_origin):
    print(process[0] + ' - ' + file_origin[0])

    if process[0] == 'wordsemb':
        call_wordsemb(file_origin)
        


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