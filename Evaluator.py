import ast
import csv
from matplotlib import pyplot
from collections import defaultdict
from os.path import basename

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

    points = {'Recall': recall_points, 'Precision': precision_points}
    return points


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

    map = sum_average_precision/len(found_results)

    return map


class Evaluator(Module):
    def __init__(self, config_file, stem):
        super().__init__('Evaluator Module', 'logs\Evaluator.log', stem)
        filename = basename(config_file)
        self.logger.log_start_activity('Reading Configuration File %s' % filename)

        config = self.read_configuration_file(config_file)
        self.expected_file = config.get('ESPERADOS')[0]
        self.results_file = config.get('RESULTADOS')[0]
        self.evaluations = {}

        self.pxr_stem_graph_file = config.get('PRECISION_RECALL_STEM_GRAPH')[0]
        self.pxr_stem_table_file = config.get('PRECISION_RECALL_STEM_TABLE')[0]
        self.pxr_no_stem_graph_file = config.get('PRECISION_RECALL_NO_STEM_GRAPH')[0]
        self.pxr_no_stem_table_file = config.get('PRECISION_RECALL_NO_STEM_TABLE')[0]

        self.eval_stem_table_file = config.get('EVALUATION_STEM_TABLE')[0]
        self.eval_no_stem_table_file = config.get('EVALUATION_NO_STEM_TABLE')[0]

        self.expected_results = defaultdict(list)
        self.found_results = defaultdict(list)

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
                votes = int(line['votes'])
                self.expected_results[query].append((document, votes))
        self.logger.log_ending_activity()

    def read_results(self):
        filename = basename(self.results_file)
        self.logger.log_start_activity('Reading results from file {0}'.format(filename))
        with open(self.results_file) as csv_file:
            field_names = ['query', 'results']
            reader = csv.DictReader(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)
            for line in reader:
                query = int(line['query'])
                self.found_results[query] = ast.literal_eval(line['results'])
        self.logger.log_ending_activity()

    def process_evaluations(self):
        self.logger.log_start_activity('Evaluating results')

        self.logger.log_start_activity('Evaluation by Precision x Recall (11 points)')
        self.evaluations['PxR 11 points'] = calculate_precision_recall(self.expected_results, self.found_results)
        self.logger.log_ending_activity()

        self.logger.log_start_activity('Evaluation by F1')
        self.evaluations['F1'] = calculate_f1(self.expected_results, self.found_results)
        self.logger.log_ending_activity()

        self.logger.log_start_activity('Evaluation by Mean Average Precision')
        self.evaluations['MAP'] = calculate_map(self.expected_results, self.found_results)
        self.logger.log_ending_activity()

        self.logger.log_ending_activity()

    def write_evaluations(self):
        self.logger.log_start_activity('Writing evaluation data')

        if self.use_stem:
            filename = self.eval_stem_table_file
        else:
            filename = self.eval_no_stem_table_file

        with open(filename, 'w+') as csv_file:
            field_names = ['Evaluation', 'Result']
            writer = csv.DictWriter(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)
            writer.writeheader()
            for evaluation, result in self.evaluations.items():
                writer.writerow({'Evaluation': evaluation, 'Result': result})

        self.logger.log_ending_activity()

    def write_table_graph(self):
        self.logger.log_start_activity('Writing graphs and tables')
        point_list = self.evaluations['PxR 11 points']

        if self.use_stem:
            graph_filename = self.pxr_stem_graph_file
            table_filename = self.pxr_stem_table_file
        else:
            graph_filename = self.pxr_no_stem_graph_file
            table_filename = self.pxr_no_stem_table_file

        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        pyplot.xlim(0, 1)
        pyplot.ylim(0, 1)

        pyplot.plot(point_list['Recall'], point_list['Precision'])
        pyplot.savefig(graph_filename)

        with open(table_filename, 'w+') as csv_file:
            field_names = ['Recall', 'Precision']
            writer = csv.DictWriter(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)
            writer.writeheader()
            for x_coord, y_coord in zip(point_list['Recall'], point_list['Precision']):
                writer.writerow({'Recall': x_coord, 'Precision': y_coord})

        self.logger.log_ending_activity()

    def execute(self):
        self.read_expected()
        self.read_results()
        self.process_evaluations()
        self.write_evaluations()
        self.write_table_graph()
