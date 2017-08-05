from ActivityLogger import ActivityLogger
from Indexer import Indexer
from InvertedListGenerator import InvertedListGenerator
from Module import Module
from QueryProcessor import QueryProcessor
from Searcher import Searcher


def str_to_bool(string):
    if string == 'True':
        return True
    elif string == 'False':
        return False
    else:
        raise ValueError


class App(object):
    def __init__(self):
        self.logger = ActivityLogger('BRI Exercise 1', 'logs/Main.log')
        config = Module.read_configuration_file('config/stem.cfg')
        self.stem = str_to_bool(config.get('USESTEM')[0]);

    def run_module_stem(self, module_class, config_file):
        activity = 'Running Module {0}'.format(module_class.__name__)
        self.logger.log_start_activity(activity)
        module_obj = module_class(config_file, True)
        module_obj.execute()
        self.logger.log_ending_activity()

    def run_module(self, module_class, config_file):
        activity = 'Running Module {0}'.format(module_class.__name__)
        self.logger.log_start_activity(activity)
        module_obj = module_class(config_file)
        module_obj.execute()
        self.logger.log_ending_activity()

    def execute(self):
        self.logger.log_start_activity('BRI Exercise 2')
        self.logger.log_stem_system(self.stem)

        self.run_module(QueryProcessor, 'config/pc.cfg')

        if self.stem:
            self.run_module_stem(InvertedListGenerator, 'config/gli.cfg')
        else:
            self.run_module(InvertedListGenerator, 'config/gli.cfg')

        self.run_module(Indexer, 'config/index.cfg')

        if self.stem:
            self.run_module_stem(Searcher, 'config/busca.cfg')
        else:
            self.run_module(Searcher, 'config/busca.cfg')

        self.logger.log_ending_activity()


app = App()
app.execute()
