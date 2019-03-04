
import argparse
#import os
#import traceback
#import logging
#import time

import nltk

from controller.selector import *





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--process', type=str, nargs='+')
    parser.add_argument('--file_origin', type=str, nargs='+')

    parser.add_argument('--file_folder', type=str, default='../input')
    parser.add_argument('--log_folder', type=str, default='../log')
    parser.add_argument('--config_folder', type=str, default='../config')
    
    args = parser.parse_args()

    print(args)

    ####
    #### TODO: set log-file

    process_selection(args.file_folder
                        , args.process
                        , args.file_origin)
