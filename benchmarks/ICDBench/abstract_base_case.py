from abc import ABC, abstractmethod


class AbstractBaseCase(ABC):

    def sample_data(self):
        pass

    def downstream_code(self):
        pass

    def assumption_in_natural_language(self):
        pass

    def target_column(self):
        pass

    def ground_truth_constraint(self):
        pass

    def data_to_pass(self):
        pass

    def data_to_reject(self):
        pass