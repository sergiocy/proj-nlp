
import argparse
import os
# import traceback
import logging
import time

import nltk




def process_selection(process, file_origin):
    print(process[0] + ' - ' + file_origin[0])

    if process[0] == 'wordsemb':
        try:
            print(file_origin[0])
            f = open (file_origin[0], encoding = 'utf8')
            txt = f.read()
            print(f.read())

            words=nltk.word_tokenize(f.read())
            words=[word.lower() for word in words]
            print(words)
            print(len(words))

        except Exception as e:
            print(e)




if __name__ == "__main__":
    print('test')

    parser = argparse.ArgumentParser()
    parser.add_argument('--process', type=str, nargs='+')
    parser.add_argument('--file_origin', type=str, nargs='+')
    args = parser.parse_args()

    print(args.process, args.file_origin)
    process_selection(args.process, args.file_origin)
