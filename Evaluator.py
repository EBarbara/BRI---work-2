import ast
import csv

import math
from matplotlib import pyplot
from collections import defaultdict
from os.path import basename

from nltk import OrderedDict

from Module import Module


def calculate_precision_recall(expected_results, found_results):
    recall_points = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    full_precision_data = []
    for query in found_results:
        query_found = found_results[query]
        query_expected = expected_results[query]

        query_expected_documents = []
        for expected_data in query_expected:
            document_number = expected_data[0]
            query_expected_documents.append(document_number)

        relevant_documents = 0
        precision_per_recall = {}
        for found_data in query_found:
            document_count = found_data[0]
            document_number = found_data[1]
            if document_number in query_expected_documents:
                relevant_documents += 1
                recall = float(relevant_documents) / len(query_expected)
                precision = float(relevant_documents) / document_count
                precision_per_recall[recall] = precision

        # Getting all ranked precisions and picking the highest
        precisions_of_query = []
        for recall_point in recall_points:
            precisions_on_recall = []
            for recall_value in precision_per_recall:
                if recall_value < recall_point:
                    continue
                precisions_on_recall.append(precision_per_recall[recall_value])

            if precisions_on_recall:
                precisions_of_query.append(max(precisions_on_recall))
            elif precisions_of_query:
                precisions_of_query.append(precisions_of_query[-1])
            else:
                precisions_of_query.append(0)

        # Grouping all query precisions in a list of lists
        full_precision_data.append(precisions_of_query)

    # Getting the average of precisions (each by query) in a given recall
    precision_points = []
    for i in range(0, 11):
        point_sum = 0
        for precisions_of_query in full_precision_data:
            point_sum += precisions_of_query[i]
        point_value = float(point_sum) / len(found_results)
        precision_points.append(point_value)

    return {'Recall': recall_points, 'Precision': precision_points}


def calculate_f1(expected_results, found_results):
    f1_sum = 0

    # Calculate the measure by query
    for query in found_results:
        query_found = found_results[query]
        query_expected = expected_results[query]

        query_expected_documents = []
        for expected_data in query_expected:
            document_number = expected_data[0]
            query_expected_documents.append(document_number)
        relevant_documents = 0
        for found_data in query_found:
            document_number = found_data[1]
            if document_number in query_expected_documents:
                relevant_documents += 1

        recall = float(relevant_documents) / len(query_expected)
        precision = float(relevant_documents) / len(query_found)
        f1_sum += (2 * recall * precision) / (recall + precision)

    # Getting the average of f1 by query
    return f1_sum / len(found_results)


def calculate_map(expected_results, found_results):
    sum_average_precision = 0
    for query in found_results:
        query_found = found_results[query]
        query_expected = expected_results[query]

        query_expected_documents = []
        for expected_data in query_expected:
            document_number = expected_data[0]
            query_expected_documents.append(document_number)

        relevant_documents = 0
        precisions = []
        for found_data in query_found:
            document_count = found_data[0]
            document_number = found_data[1]
            if document_number in query_expected_documents:
                relevant_documents += 1
                precision = float(relevant_documents) / document_count
                precisions.append(precision)

        sum_precisions = 0
        for precision in precisions:
            sum_precisions += precision
        sum_average_precision += float(sum_precisions) / len(precisions)

    return sum_average_precision / len(found_results)


def calculate_precision_at_k(expected_results, found_results, k):
    sum_precision_k = 0

    # Calculate the measure by query
    for query in found_results:
        query_found = found_results[query]
        query_expected = expected_results[query]

        query_expected_documents = []
        for expected_data in query_expected:
            document_number = expected_data[0]
            query_expected_documents.append(document_number)

        relevant_documents = 0
        for found_data in query_found[:k]:
            document_number = found_data[1]
            if document_number in query_expected_documents:
                relevant_documents += 1
        precision = float(relevant_documents) / k
        sum_precision_k += precision

    # Getting the average of precision at k by query
    return sum_precision_k / len(found_results)


def calculate_r_precision(expected_results, found_results):
    query_data = []
    precision_data = []

    # Calculate the measure by query
    for query in found_results:
        query_data.append(query)

        query_found = found_results[query]
        query_expected = expected_results[query]
        r = len(query_expected)

        query_expected_documents = []
        for expected_data in query_expected:
            document_number = expected_data[0]
            query_expected_documents.append(document_number)

        relevant_documents = 0
        for found_data in query_found[:r]:
            document_number = found_data[1]
            if document_number in query_expected_documents:
                relevant_documents += 1
        precision = float(relevant_documents) / r
        precision_data.append(precision)

    return {'Query': query_data, 'Precision': precision_data}


def calculate_mrr(expected_results, found_results):
    sum_inverse_rank = 0
    num_queries = len(found_results)

    for query in found_results:
        query_found = found_results[query]
        query_expected = expected_results[query]

        query_expected_documents = []
        for expected_data in query_expected:
            document_number = expected_data[0]
            query_expected_documents.append(document_number)

        for found_data in query_found:
            document_rank = found_data[0]
            document_number = found_data[1]
            if document_number in query_expected_documents:
                sum_inverse_rank += 1 / document_rank
                break

    mrr = (1 / num_queries) * sum_inverse_rank
    return mrr


def calculate_ndcg(expected_results, found_results):
    sum_ndcg = 0

    # Calculate the measure by query
    for query in found_results:
        query_found = found_results[query]
        query_expected = expected_results[query]

        relevant_documents = {}
        for expected_data in query_expected:
            document_number = expected_data[0]
            document_score = expected_data[1]
            relevant_documents[document_number] = document_score

        # Calculating DCG
        dcg = 0
        for found_data in query_found:
            document_rank = found_data[0]
            document_number = found_data[1]
            if document_number in relevant_documents:
                # Doesn't make difference if not - a document score = 0 would turn all the sum element = 0
                document_score = relevant_documents[document_number]
                dcg += ((2 ** document_score) - 1) / (math.log(document_rank + 1, 2))

        # Calculating IDCG
        expected_by_relevance = OrderedDict(sorted(relevant_documents.items(), key=lambda t: t[1], reverse=True))

        idcg = 0
        for index, document in enumerate(expected_by_relevance, start=1):
            score = expected_by_relevance[document]
            idcg += ((2 ** score) - 1) / (math.log(index + 1, 2))

        if idcg:  # just to avoid divide by zero
            sum_ndcg += dcg / idcg

    # Getting the average of n-dcg by query
    return sum_ndcg / len(found_results)


def calculate_bpref(expected_results, found_results):
    sum_bpref = 0

    # Calculate the measure by query
    for query in found_results:
        query_found = found_results[query]
        query_expected = expected_results[query]

        relevant_documents = []
        for expected_data in query_expected:
            document_number = expected_data[0]
            relevant_documents.append(document_number)

        # For each document found, calculate the sommatory
        sum_data = 0
        irrelevant_documents_found = 0
        for found_data in query_found:
            document_number = found_data[1]
            if document_number in relevant_documents:
                sum_data += (1 - (min(irrelevant_documents_found, len(relevant_documents)) / len(relevant_documents)))
            else:
                irrelevant_documents_found += 1
        sum_bpref += (1 / len(relevant_documents)) * sum_data

    # Getting the average of bpref by query
    return sum_bpref / len(found_results)


class Evaluator(Module):
    def __init__(self, config_file, stem):
        super().__init__('Evaluator Module', 'logs\Evaluator.log')
        filename = basename(config_file)
        self.logger.log_start_activity('Reading Configuration File %s' % filename)

        config = self.read_configuration_file(config_file)
        self.expected_file = config.get('ESPERADOS')[0]
        self.results_file = config.get('RESULTADOS')[0]
        self.results_file_stem = config.get('RESULTADOS_STEM')[0]
        self.evaluations = {}
        self.pxr_table = {}
        self.r_table = {}

        self.pxr_graph_file = config.get('PRECISION_RECALL_GRAPH')[0]
        self.pxr_table_file = config.get('PRECISION_RECALL_TABLE')[0]
        self.r_graph_file = config.get('P_R_HISTOGRAM_GRAPH')[0]
        self.r_table_file = config.get('P_R_HISTOGRAM_TABLE')[0]
        self.eval_table_file = config.get('EVALUATION_TABLE')[0]

        self.stem_key = 'STEM'
        self.nostem_key = 'NO STEM'

        self.expected_results = defaultdict(list)
        self.found_results = defaultdict(list)
        self.found_results_stem = defaultdict(list)

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
                votes = int(line['votes'])
                self.expected_results[query].append((document, votes))
        self.logger.log_ending_activity()

    def read_results(self, results_file):
        filename = basename(results_file)
        self.logger.log_start_activity('Reading results from file {0}'.format(filename))

        results = defaultdict(list)
        with open(results_file) as csv_file:
            field_names = ['query', 'results']
            reader = csv.DictReader(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)
            for line in reader:
                query = int(line['query'])
                results[query] = ast.literal_eval(line['results'])
        self.logger.log_ending_activity()
        return results

    def process_evaluations(self):
        self.logger.log_start_activity('Evaluating results')

        self.logger.log_start_activity('Evaluation by Precision x Recall (11 points)')
        self.pxr_table[self.nostem_key] = calculate_precision_recall(self.expected_results, self.found_results)
        self.pxr_table[self.stem_key] = calculate_precision_recall(self.expected_results, self.found_results_stem)
        self.logger.log_ending_activity()

        self.logger.log_start_activity('Evaluation by F1')
        self.evaluations['F1'] = {}
        self.evaluations['F1'][self.nostem_key] = calculate_f1(self.expected_results, self.found_results)
        self.evaluations['F1'][self.stem_key] = calculate_f1(self.expected_results, self.found_results_stem)
        self.logger.log_ending_activity()

        self.logger.log_start_activity('Evaluation by Mean Average Precision')
        self.evaluations['MAP'] = {}
        self.evaluations['MAP'][self.nostem_key] = calculate_map(self.expected_results, self.found_results)
        self.evaluations['MAP'][self.stem_key] = calculate_map(self.expected_results, self.found_results_stem)
        self.logger.log_ending_activity()

        self.logger.log_start_activity('Evaluation by Precision at 5')
        self.evaluations['P@5'] = {}
        self.evaluations['P@5'][self.nostem_key] = calculate_precision_at_k(self.expected_results,
                                                                            self.found_results, 5)
        self.evaluations['P@5'][self.stem_key] = calculate_precision_at_k(self.expected_results,
                                                                          self.found_results_stem, 5)
        self.logger.log_ending_activity()

        self.logger.log_start_activity('Evaluation by Precision at 10')
        self.evaluations['P@10'] = {}
        self.evaluations['P@10'][self.nostem_key] = calculate_precision_at_k(self.expected_results,
                                                                             self.found_results, 10)
        self.evaluations['P@10'][self.stem_key] = calculate_precision_at_k(self.expected_results,
                                                                           self.found_results_stem, 10)
        self.logger.log_ending_activity()

        self.logger.log_start_activity('Evaluation by P-R(A,B)')
        self.r_table[self.nostem_key] = calculate_r_precision(self.expected_results, self.found_results)
        self.r_table[self.stem_key] = calculate_r_precision(self.expected_results, self.found_results_stem)
        self.logger.log_ending_activity()

        self.logger.log_start_activity('Evaluation by Mean Reciprocal Rank')
        self.evaluations['MRR'] = {}
        self.evaluations['MRR'][self.nostem_key] = calculate_mrr(self.expected_results, self.found_results)
        self.evaluations['MRR'][self.stem_key] = calculate_mrr(self.expected_results, self.found_results_stem)
        self.logger.log_ending_activity()

        self.logger.log_start_activity('Evaluation by Normalized Discount Cumulative Gain')
        self.evaluations['N-DCG'] = {}
        self.evaluations['N-DCG'][self.nostem_key] = calculate_ndcg(self.expected_results, self.found_results)
        self.evaluations['N-DCG'][self.stem_key] = calculate_ndcg(self.expected_results, self.found_results_stem)
        self.logger.log_ending_activity()

        self.logger.log_start_activity('Evaluation by BPREF')
        self.evaluations['BPREF'] = {}
        self.evaluations['BPREF'][self.nostem_key] = calculate_bpref(self.expected_results, self.found_results)
        self.evaluations['BPREF'][self.stem_key] = calculate_bpref(self.expected_results, self.found_results_stem)
        self.logger.log_ending_activity()

        self.logger.log_ending_activity()

    def write_evaluations(self):
        self.logger.log_start_activity('Writing evaluation data')

        filename = self.eval_table_file
        with open(filename, 'w+') as csv_file:
            field_names = ['Evaluation', 'Not using Stemmer', 'Using Stemmer']
            writer = csv.DictWriter(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)
            writer.writeheader()
            for evaluation, result in self.evaluations.items():
                writer.writerow({'Evaluation': evaluation,
                                 'Not using Stemmer': result[self.nostem_key],
                                 'Using Stemmer': result[self.stem_key]})

        self.logger.log_ending_activity()

    def write_pxr_table_graph(self):
        self.logger.log_start_activity('Writing PxR graph and table')

        point_list_no_stem = self.pxr_table[self.nostem_key]
        point_list_stem = self.pxr_table[self.stem_key]

        graph_filename = self.pxr_graph_file
        table_filename = self.pxr_table_file

        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        pyplot.xlim(0, 1)
        pyplot.ylim(0, 1)

        no_stem_line = pyplot.plot(point_list_no_stem['Recall'], point_list_no_stem['Precision'],
                                   color="red", label="Not using Stemmer")
        stem_line = pyplot.plot(point_list_stem['Recall'], point_list_stem['Precision'],
                                color="blue", label="Using Stemmer")
        pyplot.legend(handles=[no_stem_line[0], stem_line[0]])
        pyplot.savefig(graph_filename)

        with open(table_filename, 'w+') as csv_file:
            field_names = ['Recall', 'Precision with Stemmer', 'Precision without Stemmer']
            writer = csv.DictWriter(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)
            writer.writeheader()
            for recall, precision_no_stem, precision_stem in zip(
                    point_list_no_stem['Recall'], point_list_no_stem['Precision'], point_list_stem['Precision']):
                writer.writerow({'Recall': recall,
                                 'Precision with Stemmer': precision_stem,
                                 'Precision without Stemmer': precision_no_stem})

        self.logger.log_ending_activity()

    def write_r_table_graph(self):
        self.logger.log_start_activity('Writing P-R(A,B) graph and table')

        point_list_no_stem = self.r_table[self.nostem_key]
        point_list_stem = self.r_table[self.stem_key]

        graph_filename = self.r_graph_file
        table_filename = self.r_table_file

        pyplot.clf()
        pyplot.xlabel('Query')
        pyplot.ylabel('A-B')
        pyplot.xlim(0, 100)
        pyplot.ylim(-0.8, 0.8)

        point_list = {'Query': [], 'Result': []}

        for query, precision_no_stem, precision_stem in zip(
                point_list_no_stem['Query'], point_list_no_stem['Precision'], point_list_stem['Precision']):
            point_list['Query'].append(query)
            point_list['Result'].append(precision_stem - precision_no_stem)

        pyplot.bar(point_list['Query'], point_list['Result'])
        pyplot.savefig(graph_filename)

        with open(table_filename, 'w+') as csv_file:
            field_names = ['Query', 'Precision with Stemmer', 'Precision without Stemmer', 'Comparison A-B']
            writer = csv.DictWriter(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)
            writer.writeheader()
            for query, precision_no_stem, precision_stem in zip(
                    point_list_no_stem['Query'], point_list_no_stem['Precision'], point_list_stem['Precision']):
                writer.writerow({'Query': query,
                                 'Precision with Stemmer': precision_stem,
                                 'Precision without Stemmer': precision_no_stem,
                                 'Comparison A-B': precision_stem - precision_no_stem})

        self.logger.log_ending_activity()

    def execute(self):
        self.read_expected()
        self.found_results = self.read_results(self.results_file)
        self.found_results_stem = self.read_results(self.results_file_stem)
        self.process_evaluations()
        self.write_evaluations()
        self.write_pxr_table_graph()
        self.write_r_table_graph()
