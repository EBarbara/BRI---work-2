import logging
import time


class ActivityLogger:
    def __init__(self, module_name, log_file):
        self.logger = logging.getLogger(module_name)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.module_name = module_name
        self.activities = []

    def log_start_activity(self, activity):
        activity_start = time.time()
        self.activities.append((activity, activity_start))
        self.logger.info('{0} - Starting {1}'.format(self.module_name, activity))

    def log_ending_activity(self):
        activity = self.activities.pop()
        elapsed_time = time.time() - activity[1]
        self.logger.info('{0} - Finishing {1}: {2:.4f}s'.format(self.module_name, activity[0], elapsed_time))

    def log_ending_activity_averaged(self, slice_name, slice_count):
        activity = self.activities.pop()
        elapsed_time = time.time() - activity[1]
        average_time = elapsed_time / slice_count
        self.logger.info('{0} - Finishing {1}: {2:.4f}s, with an average of {3:.4f}s per {4}'
                         .format(self.module_name, activity[0], elapsed_time, average_time, slice_name))

    def log_info(self, message):
        self.logger.info('{0} - {1}'.format(self.module_name, message))

    def log_warn(self, message):
        self.logger.warning('{0} - {1}'.format(self.module_name, message))
