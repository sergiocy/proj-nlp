import argparse
import logging
import os
import traceback
import time

#### OLD imports
# from controller.selector import process_selection
# from controller.InputArguments import InputArguments
from app.lib.py.local.remove_file_if_exists import remove_file_if_exists
from app.lib.py.logging.create_logger import create_logger
from app.classes.EmbeddingDataFrame import EmbeddingDataFrame



#### variables from config or input arguments
PATH_CONFIG = '../config/config.ini'
PATH_LOG_FILE = 'log.log'

PATH_SENTENCES_CSV_FILE = '/var/scdata/nlp/input/wordsim353/combined-definitions-context.csv'
PATH_SENTENCES_PROCESSED_CSV_FILE = 'tmp/textin1.csv.gz'
PATH_EMBEDDING_CSV_FILE = 'tmp/textemb1.csv.gz'

PATH_W2V_MODEL = '/var/scmodel/w2v/GoogleNews-vectors-negative300.bin'



if __name__ == "__main__":
    print(PATH_LOG_FILE)
    print(PATH_CONFIG)
    print(PATH_EMBEDDING_CSV_FILE)
    print(PATH_SENTENCES_CSV_FILE)

    start = time.time()
    remove_file_if_exists(PATH_LOG_FILE)
    logger = create_logger(PATH_LOG_FILE)
    logger.info(' - starting execution')

    obj_emb = EmbeddingDataFrame(logger=logger
                                 , filepath=PATH_SENTENCES_CSV_FILE
                                 , filepath_prepared_text=PATH_SENTENCES_PROCESSED_CSV_FILE
                                 , colname_text=['context'] #['word', 'definition', 'context']
                                 , filepath_embeddings=PATH_EMBEDDING_CSV_FILE

                                 #### IF build==True; 2 files generated (prepared texts)
                                 #### ELSE; read an embeddings file with EmbeddingDataFrame structure
                                 , build=True
                                 , type_model='W2V'
                                 , path_model=PATH_W2V_MODEL
                                 , dim_embedding=300
                                 , root_name_vect_cols='dim_')
    obj_emb.get_embeddings()
