
import argparse
import logging
#import os
#import traceback
#import time

from controller.selector import *
from controller.InputArguments import InputArguments


PATH_CONFIG = '../config/config.ini'




if __name__ == "__main__":

    obj_config = InputArguments(PATH_CONFIG, type_config_json = False)
    param_config = obj_config.get_input_arguments()
    print(param_config)
    logging.info(param_config)
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('--process', type=str, nargs='+', default=param_config.get('process'))
    parser.add_argument('--process', type=str, default=param_config.get('process'))
    parser.add_argument('--file_origin', type=str, default=param_config.get('file_origin'))
    
    parser.add_argument('--file_folder', type=str, default=param_config.get('file_folder'))
    parser.add_argument('--log_folder', type=str, default=param_config.get('log_folder'))

    parser.add_argument('--txt_lcase', type=str, default=param_config.get('log_folder'))
    parser.add_argument('--txt_tokenize', type=str, default=param_config.get('log_folder'))

    args = parser.parse_args()
    print(args)

    ####
    #### TODO: set log-file


    process_selection(param_config)
