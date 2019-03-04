
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
    parser.add_argument('--file_origin_folder', type=str, default='../input')
    args = parser.parse_args()

    print(args)

    process_selection(args.file_origin_folder, args.process, args.file_origin)
