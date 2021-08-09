
import argparse
import logging
#import os
#import traceback
#import time

from controller.selector import process_selection
from controller.InputArguments import InputArguments


PATH_CONFIG = '../config/config.ini'



if __name__ == "__main__":

    ####
    #### get config parameters
    obj_config = InputArguments(PATH_CONFIG, type_config_json = False)
    param_config = obj_config.get_input_arguments()

    parser = argparse.ArgumentParser()
    #parser.add_argument('--process', type=str, nargs='+', default=param_config.get('process'))
    parser.add_argument('--process', type=str, default=param_config.get('process'))
    parser.add_argument('--file_origin', type=str, default=param_config.get('file_origin'))
    parser.add_argument('--file_folder', type=str, default=param_config.get('file_folder'))
    parser.add_argument('--file_csv', type=str, default=param_config.get('file_csv'))
    parser.add_argument('--log_folder', type=str, default=param_config.get('log_folder'))
    parser.add_argument('--log_file', type=str, default=param_config.get('log_file'))
    parser.add_argument('--txt_lcase', type=str, default=param_config.get('txt_lcase'))
    parser.add_argument('--txt_tokenize', type=str, default=param_config.get('txt_tokenize'))
    args = parser.parse_args()

    ####
    #### set log-file
    logging.basicConfig(filename=param_config.get('log_folder')+param_config.get('log_file')
                        , level=logging.DEBUG
                        , filemode='w'
                        , format='%(asctime)s - %(levelname)s - %(message)s'
                        )

    #####
    ##### set/overwrite in dictionary param_config the parameters getted by command line
    logging.info('config file parameters: ' + str(param_config))

    obj_config.set_input_args_from_cli(args)
    param_config = obj_config.get_input_arguments()

    logging.info('resetting parameters from command line: ' + str(param_config))

    #####
    ##### calling function to define and parametrize the process selected
    process_selection(param_config)
