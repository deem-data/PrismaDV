from .abstract_base_case import AbstractBaseCase


class KnowledgeInSQLSubtractColumnLog(AbstractBaseCase):

    def sample_data(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "profit": [100, 100, 1000, 10000],
            "cost": [9, 9, 99, 99],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }

    def downstream_code(self):
        return """
        df_processed = duckdb.sql('''
          SELECT profit - cost AS profit_after_cost
          FROM df
          WHERE profit IS NOT NULL OR cost is NOT NULL
        ''').df()
        df_processed['profit_after_cost_smoothed'] = df_processed['profit_after_cost'].apply(math.log)
        """

    def assumption_in_natural_language(self):
        return "All values in the profit column should be greater than the values in the cost column."

    def target_column(self):
        return "profit"

    def ground_truth_constraint(self):
        return ".isGreaterThan('profit', 'cost')"

    def data_to_pass(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "profit": [10, 1000, 2000, 10000],
            "cost": [9, 9, 1999, 99],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }

    def data_to_reject(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "profit": [100, 100, 1000, 10000],
            "cost": [9, 9, 1000, 99],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }
