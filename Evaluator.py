import ast
import csv
from matplotlib import pyplot
from collections import defaultdict
from os.path import basename

from Module import Module


def write_graph(point_list, x_label, y_label, filename):
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.xlim(0, 1)
    pyplot.ylim(0, 1)

    pyplot.plot(point_list['x'], point_list['y'])
    pyplot.savefig(filename)


def write_table(point_list, x_label, y_label, filename):
    with open(filename, 'w+') as csv_file:
        field_names = [x_label, y_label]
        writer = csv.DictWriter(csv_file, delimiter=';', lineterminator='\n', fieldnames=field_names)
        writer.writeheader()
        for x_coord, y_coord in zip(point_list['x'], point_list['y']):
            writer.writerow({x_label: x_coord, y_label: y_coord})


class Evaluator(Module):
    def __init__(self, config_file, stem):
        super().__init__('Evaluator Module', 'logs\Evaluator.log', stem)
        filename = basename(config_file)
        self.logger.log_start_activity('Reading Configuration File %s' % filename)

        config = self.read_configuration_file(config_file)
        self.expected_file = config.get('ESPERADOS')[0]
        self.results_file = config.get('RESULTADOS')[0]

        self.pxr_stem_graph_file = config.get('PRECISION_RECALL_STEM_GRAPH')[0]
        self.pxr_stem_table_file = config.get('PRECISION_RECALL_STEM_TABLE')[0]
        self.pxr_no_stem_graph_file = config.get('PRECISION_RECALL_NO_STEM_GRAPH')[0]
        self.pxr_no_stem_table_file = config.get('PRECISION_RECALL_NO_STEM_TABLE')[0]

        self.expected_results = defaultdict(list)
        self.found_results = defaultdict(list)

        self.logger.log_stem_use(self.use_stem)
        self.logger.log_ending_activity()

    def calculate_precision_recall(self):
        self.logger.log_start_activity('Evaluation by Precision x Recall (11 points)')

        recall_points = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        full_precision_data = []
        for query in self.found_results:
            query_found = self.found_results[query]
            query_expected = self.expected_results[query]

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
            point_value = float(point_sum) / len(self.found_results)
            precision_points.append(point_value)

        points = {'x': recall_points, 'y': precision_points}

        if self.use_stem:
            pxr_graph_filename = self.pxr_stem_graph_file
            pxr_table_filename = self.pxr_stem_table_file
        else:
            pxr_graph_filename = self.pxr_no_stem_graph_file
            pxr_table_filename = self.pxr_no_stem_table_file
        write_graph(points, 'Recall', 'Precision', pxr_graph_filename)
        write_table(points, 'Recall', 'Precision', pxr_table_filename)

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
        self.calculate_precision_recall()
        self.logger.log_ending_activity()

    def execute(self):
        self.read_expected()
        self.read_results()
        self.process_evaluations()
