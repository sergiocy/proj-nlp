
import json
import configparser




#### TODO ...maybe methods to validate input arguments...




class InputArguments():

    #list_type_config = ['json', 'ini']
    input_arguments = None
    type_config_json = False
    
    
    def __init__(self, path_file_config, type_config_json = False):
        if type_config_json == True:
            self.input_arguments = self.get_json_config_parameters(path_file_config)
        else:
            self.input_arguments = self.get_ini_config_parameters(path_file_config)




    def get_input_arguments(self):
        return self.input_arguments


    
    def set_input_args_from_cli(self, input_args):
        for arg in vars(input_args):
            self.input_arguments.update({arg: getattr(input_args, arg)})
            #print(arg + ' - ' + str(getattr(input_args, arg)))

    def build_arguments_dictionary(self, config):
        dct_arguments = {
            "file_folder": config['APP_CONFIG']['PATH_FILE_FOLDER']
            , "file_origin": config['APP_CONFIG']['FILE_INPUT']
            , "log_folder": config['APP_CONFIG']['PATH_LOG_FOLDER']
            , "log_file": config['APP_CONFIG']['FILE_LOG']
            , "process": config['EXECUTION_CONFIG']['TYPE_PROCESS']
            , "txt_lcase": self.set_boolean_param(config['INPUT_TEXT_PROCESSING']['LCASE'])
            , "txt_tokenize": self.set_boolean_param(config['INPUT_TEXT_PROCESSING']['TOKENIZE'])
            }
        return dct_arguments
        

    def set_boolean_param(self, param_value):
        #### ...podeos poner excepcion para controlar que entre solo 'T' o 'F'
        if param_value is 'T':
            param_value = True
        if param_value is 'F':
            param_value = False

        return param_value


    def get_json_config_parameters(self, path_file_config):
        config = json.load(open(path_file_config, 'r'))
        dct_args = self.build_arguments_dictionary(config)
        return dct_args

    def get_ini_config_parameters(self, path_file_config):
        config = configparser.ConfigParser()
        config.read(path_file_config)
        dct_args = self.build_arguments_dictionary(config)
        return dct_args
    
    
 