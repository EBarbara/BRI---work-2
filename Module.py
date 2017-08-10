from collections import defaultdict

from ActivityLogger import ActivityLogger


class Module(object):
    def __init__(self, module_name, log_file):
        super(Module, self).__init__()
        self.module_name = module_name
        self.logger = ActivityLogger(module_name, log_file)

    '''
    Eu tentei usar o módulo configparser, mas não encontrei nenhuma forma de fazê-lo 
    trabalhar com listas de opções em Python3 (no gli.cfg, temos vários arquivos de 
    entrada). Assim, preferi criar meu próprio leitor de configurações
    '''

    @staticmethod
    def read_configuration_file(file):
        config = defaultdict(list)
        with open(file) as config_file:
            for line in config_file:
                data = line.rstrip().split('=')
                config[data[0]].append(data[1])
        return config
