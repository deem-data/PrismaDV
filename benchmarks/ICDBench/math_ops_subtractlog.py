from .abstract_base_case import AbstractBaseCase


class MathOpsSubtractLog(AbstractBaseCase):

    def sample_data(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "cost": [100, 100, 1000, 1000],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }

    def downstream_code(self):
        return """
        df['cost_adjusted'] = df['cost'] - 23
        df['cost_smoothed'] = df['cost_adjusted'].apply(math.log)
        """

    def assumption_in_natural_language(self):
        return "All values in the cost column should be greater than 23."

    def target_column(self):
        return "cost"

    def ground_truth_constraint(self):
        return ".satisfies('cost > 23', 'cost must be greater than 23')"

    def data_to_pass(self):
        return {
            "booking_id": [5, 6, 7, 8],
            "cost": [24, 1000, 120, 1100],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }

    def data_to_reject(self):
        return {
            "booking_id": [5, 6, 7, 8],
            "cost": [100, 0, 21, 10],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }
