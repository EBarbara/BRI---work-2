import csv
from os.path import basename

from lxml import etree

from Module import Module


def calculate_votes(score):
    result = 0
    while score:
        result += score % 10
        score //= 10
    return result


class QueryProcessor(Module):
    def __init__(self, config_file):
        super().__init__('Query Processor Module', 'logs\QueryProcessor.log')
        filename = basename(config_file)
        self.logger.log_start_activity('Reading Configuration File %s' % filename)

        config = self.read_configuration_file(config_file)
        self.input_file = config.get('LEIA')[0]
        self.processed_queries_file = config.get('CONSULTAS')[0]
        self.expected_results_file = config.get('ESPERADOS')[0]
        self.raw_queries = None
        self.processed_queries = {}
        self.logger.log_ending_activity()

    def read_raw_queries(self):
        filename = basename(self.input_file)
        self.logger.log_start_activity('Reading raw query data from file {0}'.format(filename))
        parser = etree.XMLParser(dtd_validation=True)
        self.raw_queries = etree.parse(self.input_file, parser)
        self.logger.log_ending_activity()

    # Acho que reinventei a roda - esse dicionário de dicionários, quando impresso vira basicamente um JSON
    def process_queries(self):
        self.logger.log_start_activity('Processing queries')
        for raw_query in self.raw_queries.getroot().iterchildren():
            for element in raw_query.iterchildren():
                if element.tag == 'QueryNumber':
                    query_number = int(element.text)
                    self.processed_queries[query_number] = {}
                elif element.tag == 'QueryText':
                    query_text = element.text.replace('\n  ', '').replace('\n', '')
                    self.processed_queries[query_number]['text'] = query_text.upper()
                elif element.tag == 'Records':
                    self.processed_queries[query_number]['results'] = {}
                    for item in element.iterchildren():
                        document = int(item.text)
                        votes = calculate_votes(int(item.attrib.get("score")))
                        self.processed_queries[query_number]['results'][document] = calculate_votes(votes)
        self.logger.log_ending_activity()

    def write_processed_queries(self):
        filename = basename(self.processed_queries_file)
        self.logger.log_start_activity('Writing Processed Queries File {0}'.format(filename))
        with open(self.processed_queries_file, 'w+') as csv_file:
            field_names = ['query', 'text']
            writer = csv.DictWriter(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)
            for query in self.processed_queries:
                writer.writerow({'query': query, 'text': self.processed_queries[query]['text']})
        self.logger.log_ending_activity()

    def write_expected_results(self):
        filename = basename(self.expected_results_file)
        self.logger.log_start_activity('Writing Expected Results File {0}'.format(filename))
        with open(self.expected_results_file, 'w+') as csv_file:
            field_names = ['query', 'document', 'votes']
            writer = csv.DictWriter(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)
            for query in self.processed_queries:
                results = self.processed_queries[query]['results']
                for document in results:
                    writer.writerow({'query': query, 'document': document, 'votes': results[document]})
        self.logger.log_ending_activity()

    def execute(self):
        self.read_raw_queries()
        self.process_queries()
        self.write_processed_queries()
        self.write_expected_results()
