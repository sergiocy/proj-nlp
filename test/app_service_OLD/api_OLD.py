
import logging

import nltk
from service.input_text.InputText import InputText, InputCsvText



def process_selection(param_config):
    logging.info('process selected: ' + param_config.get('process'))
    logging.info('file selected: ' + param_config.get('file_folder') + param_config.get('file_origin'))
    
    if param_config.get('process') == 'wordsemb':
        logging.info('setting input text')
        txt_input = set_input_object(param_config)
        logging.info(txt_input)
        call_wordsemb(txt_input)


####
#### set and clean input text
def set_input_object(param_config):
    if param_config.get('file_csv') == True:
         in_text1 = InputCsvText(param_config.get('file_folder') + param_config.get('file_origin'))

    else:
        in_text1 = InputText(param_config.get('file_folder') + param_config.get('file_origin'))
        logging.info('input text: ' + in_text1.get_beggining_input_text())
        in_text1.set_output_text(lcase = param_config.get('txt_lcase')
                                , tokenize = param_config.get('txt_tokenize')
                                )

        output = in_text1.get_output_text()


####
#### call to services
def call_wordsemb(txt_input):
    print(txt_input)
    try:
        print('calling wordsemb service')
        logging.info('calling wordsemb service')
    except Exception as e:
        print(e)


def call_similarity():
    print('calling similarity service')
    logging.info('calling similarity service')


def call_parser():
    print('calling parser service')
    logging.info('calling parser service')

    


if __name__ == "__main__":
    print('using selector.py') 