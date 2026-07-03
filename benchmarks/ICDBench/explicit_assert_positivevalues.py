from .abstract_base_case import AbstractBaseCase


class ExplicitAssertPositiveValues(AbstractBaseCase):

    def sample_data(self):
        return {
            "booking_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "cost": [1, 10, 10, 100, 1, 10, 100, 1000, 1, 1]
        }

    def downstream_code(self):
        return """
        df = df.dropna(subset=["cost"])
        assert (df["cost"] <= 0).sum() == 0
        """

    def assumption_in_natural_language(self):
        return "The cost column should only contain positive values (or null values)."

    def target_column(self):
        return "cost"

    def ground_truth_constraint(self):
        return ".isPositive('cost')"

    def data_to_pass(self):
        return {
            "booking_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "cost": [1, 10, 10, 100, 1, 10, 100, None, 1, 1, 1, 10, 10, 100, 1, 10, 100, None, 1, 1]
        }

    def data_to_reject(self):
        return {
            "booking_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "cost": [1, 10, 10, None, 1, 10, 100, None, 1, 1, 0, 10, 10, 100, 1, 10, 100, None, 1, 1]
        }
