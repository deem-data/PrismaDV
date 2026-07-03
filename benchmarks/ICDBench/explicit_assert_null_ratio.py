from .abstract_base_case import AbstractBaseCase


class ExplicitAssertNullRatio(AbstractBaseCase):

    def sample_data(self):
        return {
            "booking_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "cost": [1, 10, 10, 100, 1, 10, 100, None, 1, 1]
        }

    def downstream_code(self):
        return """
        null_count = df["cost"].isnull().sum()
        null_ratio = null_count / len(df)
        assert null_ratio <= 0.1
        """

    def assumption_in_natural_language(self):
        return "The cost column should have at most 10% null values."

    def target_column(self):
        return "cost"

    def ground_truth_constraint(self):
        return ".hasCompleteness('cost', lambda x: x >= 0.9)"

    def data_to_pass(self):
        return {
            "booking_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "cost": [1, 10, 10, 100, 1, 10, 100, None, 1, 1, 1, 10, 10, 100, 1, 10, 100, None, 1, 1]
        }

    def data_to_reject(self):
        return {
            "booking_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "cost": [1, 10, 10, None, 1, 10, 100, None, 1, 1, 1, 10, 10, 100, 1, 10, 100, None, 1, 1]
        }
