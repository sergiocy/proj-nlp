
import logging
import os
import time
import numpy as np
import pandas as pd



class EmbeddingDataFrame:

    #### attributes
    #### constructor
    def __init__(self, filepath=None
                        , colname_text=None
                        , build=False):

        self.filepath = filepath


        if build:
            self.colname_text = colname_text

            #### TODO: we receive a file as csv file with strings/sentences
            #df = load_input_text_csv(logger = None)


        else:
            self.colname_text = None
            #### we receive a file with embeddings
            #### we will generate Embedding objects
            print('loading')
