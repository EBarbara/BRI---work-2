import csv
from ast import literal_eval
from collections import defaultdict
from math import sqrt
from os.path import basename

from nltk import word_tokenize, OrderedDict

from Module import Module


def calculate_max(words):
    max_words = 0
    unique_words = list(set(words))
    for word in unique_words:
        if words.count(word) > max_words:
            max_words = words.count(word)
    return max_words


def calculate_query_vector_length(word_array):
    value = 0
    for word in word_array:
        component = float(word_array[word])
        if component > 0:
            value += component ** 2
    return sqrt(value)


class Searcher(Module):
    def __init__(self, config_file):
        super().__init__('Searcher Module', 'logs\Searcher.log')
        filename = basename(config_file)
        self.logger.log_start_activity('Reading Configuration File %s' % filename)

        config = self.read_configuration_file(config_file)
        self.queries_file = config.get('CONSULTAS')[0]
        self.model_file = config.get('MODELO')[0]
        self.results_file = config.get('RESULTADOS')[0]

        self.queries = defaultdict(list)
        self.model = {}
        self.model_documents = defaultdict(list)
        self.document_length = {}
        self.query_document_rank = {}

        self.logger.log_ending_activity()

    def read_model(self):
        filename = basename(self.model_file)
        self.logger.log_start_activity('Reading vector space model from file {0}'.format(filename))
        with open(self.model_file) as csv_file:
            field_names = ['word', 'data']
            reader = csv.DictReader(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)

            for line in reader:
                word = line['word']
                data = literal_eval(line['data'])
                self.model[word] = data
                for document in self.model[word][1]:
                    idf = self.model[word][0]
                    self.model_documents[document].append((word, idf))
        self.logger.log_ending_activity()

    def read_queries(self):
        filename = basename(self.queries_file)
        self.logger.log_start_activity('Reading processed queries from file {0}'.format(filename))
        with open(self.queries_file) as csv_file:
            field_names = ['query', 'words']
            reader = csv.DictReader(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)

            for line in reader:
                query = int(line['query'])
                words = word_tokenize(line['words'])

                # Eliminando as palavras irrelevantes da consulta (palavras com
                # menos de duas letras ou contendo algo diferente de letras)
                for word in words:
                    if len(word) > 2 and word.isalpha():
                        self.queries[query].append(word)
        self.logger.log_ending_activity()

    def calculate_document_vector_length(self):
        self.logger.log_start_activity('Calculating length of document vectors')
        for document in self.model_documents:
            data_vector = self.model_documents[document]
            value = 0
            for word in data_vector:
                idw_value = float(word[1])
                if idw_value > 0:
                    value = value + idw_value ** 2
            vector_length = sqrt(value)
            self.document_length[document] = vector_length
        self.logger.log_ending_activity()

    def normalize_tf_query(self, word, query):
        tf = self.queries[query].count(word)
        max_t = calculate_max(self.queries[query])
        return 0.5 + 0.5 * (tf / max_t)

    def build_query_vector(self, query):
        query_vector = {}
        unique_words = list(set(self.queries[query]))
        for word in unique_words:
            if word in self.model:
                tf_n = self.normalize_tf_query(word, query)
                idf = self.model[word][0]
                weight = tf_n * idf
                query_vector[word] = weight
        return query_vector

    def run_searches(self):
        self.logger.log_start_activity('Running Searches')
        for query in self.queries:
            self.logger.log_info('running query {0}'.format(query))
            query_vector = self.build_query_vector(query)
            query_vector_length = calculate_query_vector_length(query_vector)
            self.query_document_rank[query] = self.calculate_document_rank(query_vector, query_vector_length)
        self.logger.log_ending_activity_averaged('query', len(self.queries))

    def calculate_document_rank(self, query_vector, query_vector_length):
        words = query_vector.keys()
        documents_cos = {}
        for document in self.model_documents:
            document_vector_length = self.document_length[document]
            value = 0
            for pair in self.model_documents[document]:
                word = pair[0]
                weight = pair[1]
                if word in words:
                    value = value + (weight * query_vector[word])
            if value > 0:
                cos = value / (document_vector_length * query_vector_length)
                documents_cos[document] = cos

        ordered_doc_cos = OrderedDict(sorted(documents_cos.items(), key=lambda t: t[1], reverse=True))

        document_rank = []
        rank_position = 1
        for document in ordered_doc_cos:
            document_rank.append((rank_position, document, ordered_doc_cos[document]))
            rank_position += 1
        return document_rank

    def write_results(self):
        filename = basename(self.results_file)
        self.logger.log_start_activity('Writing Results File {0}'.format(filename))
        with open(self.results_file, 'w+') as csv_file:
            field_names = ['query', 'results']
            writer = csv.DictWriter(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)
            for query in self.query_document_rank:
                writer.writerow({'query': query, 'results': self.query_document_rank[query]})
        self.logger.log_ending_activity()

    def execute(self):
        self.read_model()
        self.read_queries()
        self.calculate_document_vector_length()
        self.run_searches()
        self.write_results()
