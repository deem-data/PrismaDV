from .abstract_base_case import AbstractBaseCase


class MathOpsLogDifference(AbstractBaseCase):

    def sample_data(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "cost": [1, 10, 10, 100],
            "reduction": [0, 3, 5, 90],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }

    def downstream_code(self):
        return """
        df['cost_smoothed'] = (df['cost'] - df['reduction']).apply(math.log)
        """

    def assumption_in_natural_language(self):
        return "All values in the cost column should be positive and greater than the respective values in the reduction column."

    def target_column(self):
        return "cost"

    def ground_truth_constraint(self):
        return ".satifies('cost > 0 AND cost > reduction')"

    def data_to_pass(self):
        return {
            "booking_id": [5, 6, 7, 8],
            "cost": [10, 1000, 10, 100],
            "reduction": [9, 999, 5, 0],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }

    def data_to_reject(self):
        return {
            "booking_id": [5, 6, 7, 8],
            "cost": [10, 0, 10, 100],
            "reduction": [0, 3, 10, 90],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }
