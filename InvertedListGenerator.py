import csv
from collections import defaultdict
from os.path import basename

from lxml import etree
from nltk import word_tokenize

from PorterStemmer import PorterStemmer
from Module import Module


class InvertedListGenerator(Module):
    def __init__(self, config_file, stem=False):
        super().__init__('Inverted List Generator Module', 'logs\InvertedListGenerator.log')
        filename = basename(config_file)
        self.logger.log_start_activity('Reading Configuration File %s' % filename)

        config = self.read_configuration_file(config_file)
        self.input = config.get('LEIA')
        self.output = config.get('ESCREVA')
        self.documents = dict()
        self.list = defaultdict(list)
        self.use_stem = stem
        self.stemmer = PorterStemmer()
        self.logger.log_stem_use(self.use_stem)

        self.logger.log_ending_activity()

    def read_documents(self):
        full_count = 0
        self.logger.log_start_activity('Reading document abstracts')
        parser = etree.XMLParser(dtd_validation=True)
        for input_file in self.input:
            count = 0
            filename = basename(input_file)
            self.logger.log_info('Reading Document File {0}'.format(filename))
            file = etree.parse(input_file, parser)
            for record in file.getroot().iterchildren():
                key = 'invalid'
                value = 'invalid'
                for element in record.iterchildren():
                    if element.tag == 'RECORDNUM':
                        key = int(element.text)
                    if element.tag == 'ABSTRACT' or element.tag == 'EXTRACT':
                        value = element.text
                if key != 'invalid':
                    if value != 'invalid':
                        self.documents[key] = word_tokenize(value)
                        count += 1
                    else:
                        self.logger.log_warn('Document {0} has no ABSTRACT nor EXTRACT'.format(key))
            self.logger.log_info("Read {0} documents from {1}".format(count, filename))
            full_count += count
        self.logger.log_info("Read {0} documents in total ".format(full_count))
        self.logger.log_ending_activity_averaged('document', full_count)

    def generate_list(self):
        self.logger.log_start_activity('Generating inverted list')

        for index, value in self.documents.items():
            for word in value:
                if self.use_stem:
                    processed_word = self.stemmer.stem(word, 0, len(word) - 1)
                    self.list[processed_word.upper()].append(index)
                else:
                    self.list[word.upper()].append(index)
        self.logger.log_ending_activity()

    def write_list(self):
        output_file = self.output[0]
        filename = basename(output_file)
        self.logger.log_start_activity('Writing Output File {0}'.format(filename))
        with open(output_file, 'w+') as csv_file:
            field_names = ['word', 'documents']
            writer = csv.DictWriter(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)
            for word in self.list:
                writer.writerow({'word': word, 'documents': self.list[word]})
        self.logger.log_ending_activity()

    def execute(self):
        self.read_documents()
        self.generate_list()
        self.write_list()
