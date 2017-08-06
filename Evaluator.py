import ast
import csv
from collections import defaultdict
from os.path import basename

from Module import Module


class Evaluator(Module):
    def __init__(self, config_file, stem):
        super().__init__('Evaluator Module', 'logs\Evaluator.log', stem)
        filename = basename(config_file)
        self.logger.log_start_activity('Reading Configuration File %s' % filename)

        config = self.read_configuration_file(config_file)
        self.expected_file = config.get('ESPERADOS')[0]
        self.results_file = config.get('RESULTADOS')[0]

        self.expected_documents = defaultdict(list)
        self.found_documents = defaultdict(list)

        self.logger.log_stem_use(self.use_stem)

        self.logger.log_ending_activity()

    def read_expected(self):
        filename = basename(self.expected_file)
        self.logger.log_start_activity('Reading expected results from file {0}'.format(filename))
        with open(self.expected_file) as csv_file:
            field_names = ['query', 'document', 'votes']
            reader = csv.DictReader(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)
            for line in reader:
                query = int(line['query'])
                document = int(line['document'])
                self.expected_documents[query].append(document)
        self.logger.log_ending_activity()

    def read_results(self):
        filename = basename(self.results_file)
        self.logger.log_start_activity('Reading results from file {0}'.format(filename))
        with open(self.results_file) as csv_file:
            field_names = ['query', 'results']
            reader = csv.DictReader(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)
            for line in reader:
                query = int(line['query'])
                results = ast.literal_eval(line['results'])
                results_unsorted = []
                for result in results[:40]:
                    #TODO avaliar se é de fato pra fazer culling
                    results_unsorted.append(result[1])
                self.found_documents[query] = sorted(results_unsorted)
        self.logger.log_ending_activity()

    def process_evaluations(self):
        self.logger.log_start_activity('Evaluating results')
        
        self.logger.log_ending_activity()

    def execute(self):
        self.read_expected()
        self.read_results()
        self.process_evaluations()
