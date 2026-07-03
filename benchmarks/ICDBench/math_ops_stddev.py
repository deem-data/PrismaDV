from .abstract_base_case import AbstractBaseCase


class MathOpsStddev(AbstractBaseCase):

    def sample_data(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "cost": [1, 10, 10, 100],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }

    def downstream_code(self):
        return """
        df['cost_normalised'] = (df['cost'] - df['cost'].mean()) / df['cost'].std()
        """

    def assumption_in_natural_language(self):
        return "The cost column should have a standard deviation greater than zero."

    def target_column(self):
        return "cost"

    def ground_truth_constraint(self):
        return ".hasStandardDeviation('cost', lambda x: x > 0)"

    def data_to_pass(self):
        return {
            "booking_id": [5, 6, 7, 8],
            "cost": [10, 1000, 10, 100],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }

    def data_to_reject(self):
        return {
            "booking_id": [5, 6, 7, 8],
            "cost": [10, 10, 10, 10],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }
