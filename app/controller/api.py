# -*- coding: utf-8 -*-

import logging
import configparser




def run_pipeline(logger = None
                , config_pipe_file = None):
    
    try:
        print('running flow')
        print('config file {}'.format(config_pipe_file))

        config_pipe = configparser.ConfigParser()
        config_pipe.read(config_pipe_file)

        print(config_pipe)
        print(config_pipe.sections())

        processes = {
                    "LOAD_INPUT_TEXT_CSV": run_load_input_text_csv,
                    "GET_EMBEDDINGS_AS_DF": run_get_embeddings_as_df
                    }

        for sec in config_pipe.sections():
            print(sec.split('.'))
            sec = sec.split('.')[1]

            


    except Exception as e:
        print(e)
        raise e


def run_load_input_text_csv():
    print('executing load_input_text_csv')

def run_get_embeddings_as_df():
    print('executing load_input_text_csv')
    


if __name__ == "__main__":
    print('using selector.py') 