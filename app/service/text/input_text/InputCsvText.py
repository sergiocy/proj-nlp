
from nltk import word_tokenize 


####
#### CLASS TO GET TEXTS THAT ARE IN A CSV FILE
#### 

class InputCsvText:

    input_file_name = None
    df_csv = None


    def __init__(self, file_name):
        print('we must implement a class to define texts in csv way')
        self.input_file_name = file_name




    #### function implemented to read n columns in a csv with different number 
    #### of columns by row
    #### ARGS input: n rows that we want to get, split fields symbol, colnames
    #### ARGS output: a pandas dataframe with ncols
    def read_n_cols_by_row(self, ncols=7, sep_field='\t', colnames=None):
        f = open(file, encoding = 'utf8')
        lst_file = []
        
        for line in f:
            #### split by tabulations
            lst_line = line.split(sep_field)
            #### get first 7 elements (there is cases with complementary columns)
            lst_line = lst_line[0:ncols]
            lst_file.append(lst_line)
            
        df = pd.DataFrame(columns=colnames, data=lst_file)
        
        return df

        


    
    