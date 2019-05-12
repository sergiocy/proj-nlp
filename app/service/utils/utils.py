# -*- coding: utf-8 -*-

import os




def remove_file(file_to_del):
    if os.path.exists(file_to_del):
        os.remove(file_to_del)
        


    
    

if __name__ == '__main__':

    print('in module input_processing')
    
 
