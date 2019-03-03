
import argparse
#import os
#import traceback
#import logging
#import time

import nltk


from controller.selector import *
#from controller.selector import process_selection




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--process', type=str, nargs='+')
    parser.add_argument('--file_origin', type=str, nargs='+')
    args = parser.parse_args()

    process_selection(args.process, args.file_origin)
