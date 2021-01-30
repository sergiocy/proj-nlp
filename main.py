
import argparse
import logging
import os
import traceback
import time


#### OLD imports
#from controller.selector import process_selection
#from controller.InputArguments import InputArguments

from app.lib.py.logging.create_logger import create_logger

from app.classes.EmbeddingDataFrame import EmbeddingDataFrame


#### variables from config or input arguments
PATH_CONFIG = '../config/config.ini'
PATH_LOG_FILE = 'log.log'
PATH_EMBEDDING_CSV_FILE = 'emb1.csv.gz'
PATH_SENTENCES_CSV_FILE = '/var/scdata/nlp/input/wordsim353/combined-definitions.csv'



if __name__ == "__main__":
    print(PATH_LOG_FILE)
    print(PATH_CONFIG)
    print(PATH_EMBEDDING_CSV_FILE)
    print(PATH_SENTENCES_CSV_FILE)

    start = time.time()
    #os.remove(PATH_LOG_FILE)
    logger = create_logger(PATH_LOG_FILE)
    logger.info(' - starting execution')


    obj_emb = EmbeddingDataFrame(filepath=PATH_SENTENCES_CSV_FILE, build=True)
