from .abstract_base_case import AbstractBaseCase


class ExplicitAssertNumDistinct(AbstractBaseCase):

    def sample_data(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }

    def downstream_code(self):
        return """
            num_statuses = df["status"].nunique()
            assert num_statuses <= 4
        """

    def assumption_in_natural_language(self):
        return "The status column should have at most 4 distinct values."

    def target_column(self):
        return "status"

    def ground_truth_constraint(self):
        return ".hasNumberOfDistinctValues('status', lambda x: x <= 4)"

    def data_to_pass(self):
        return {
            "booking_id": [1, 2, 3, 4, 5],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED", "UNDER_REVIEW"],
        }

    def data_to_reject(self):
        return {
            "booking_id": [1, 2, 3, 4, 5],
            "status": ["COMPLETED", "IN_PROGRESS", "UNKNOWN", "CANCELLED", "UNDER_REVIEW"],
        }
