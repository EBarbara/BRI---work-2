from ActivityLogger import ActivityLogger
from Indexer import Indexer
from InvertedListGenerator import InvertedListGenerator
from QueryProcessor import QueryProcessor
from Searcher import Searcher


class App(object):
    def __init__(self):
        self.logger = ActivityLogger('BRI Exercise 1', 'logs/Main.log')

    def run_module(self, module_class, config_file):
        activity = 'Running Module {0}'.format(module_class.__name__)
        self.logger.log_start_activity(activity)
        module_obj = module_class(config_file)
        module_obj.execute()
        self.logger.log_ending_activity()

    def execute(self):
        self.logger.log_start_activity('BRI Exercise 2')
        self.run_module(QueryProcessor, 'config/pc.cfg')
        self.run_module(InvertedListGenerator, 'config/gli.cfg')
        self.run_module(Indexer, 'config/index.cfg')
        self.run_module(Searcher, 'config/busca.cfg')
        self.logger.log_ending_activity()


app = App()
app.execute()
