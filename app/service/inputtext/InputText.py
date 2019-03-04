

from nltk import word_tokenize


class InputText:

    input_file_name = None
    input_text = None
    output_text = None


    def __init__(self, file_name):
        try:
            self.input_file_name = file_name
            f = open (self.input_file_name, encoding = 'utf8')
            #txt = f.read()
            self.input_text = f.read()
        except Exception as e:
            print(e)




    def get_file_name(self):
        print(self.input_file_name)  

    def get_input_text(self):
        print(self.input_text)

    def get_output_text(self):
        print(self.output_text)


    def set_output_text(self, lcase = False, tokenize = False):
        self.output_text = self.input_text
        if lcase:
            self.output_text = self.output_text.lower()    
        if tokenize:
            self.output_text = word_tokenize(self.output_text)

    