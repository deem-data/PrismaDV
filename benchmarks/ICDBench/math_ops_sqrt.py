from .abstract_base_case import AbstractBaseCase


class MathOpsSqrt(AbstractBaseCase):

    def sample_data(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "cost": [100, 100, 10000, 100],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }

    def downstream_code(self):
        return """
        hidden_cost = 10
        df['cost_smoothed'] = (df['cost'] - hidden_cost).apply(math.sqrt)
        """

    def assumption_in_natural_language(self):
        return "All values in the cost column should be greater than or equal to 10."

    def target_column(self):
        return "cost"

    def ground_truth_constraint(self):
        return ".satisfies('cost >= 10')"

    def data_to_pass(self):
        return {
            "booking_id": [5, 6, 7, 8],
            "cost": [10, 1000, 10, 100],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }

    def data_to_reject(self):
        return {
            "booking_id": [5, 6, 7, 8],
            "cost": [10, 1, 9, 100],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }
