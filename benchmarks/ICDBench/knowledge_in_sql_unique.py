from .abstract_base_case import AbstractBaseCase


class KnowledgeInSQLUnique(AbstractBaseCase):

    def sample_data(self):
        return {
            "booking_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "cost": [1, 10, 10, 100, 1, 10, 100, 1000, 1, 1]
        }

    def downstream_code(self):
        return """
        booking_details = {}
        additional_cost = 10
                        
        result = duckdb.sql(f'''
            SELECT booking_id, cost FROM df
        ''').fetchall()
        
        for row in result:
            booking_details[row[0]] = { 'real_cost': row[1] + additional_cost } 
        """

    def assumption_in_natural_language(self):
        return "The booking_id should only have unique values."

    def target_column(self):
        return "booking_id"

    def ground_truth_constraint(self):
        return ".isUnique('booking_id')"

    def data_to_pass(self):
        return {
            "booking_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "cost": [1, 10, 10, 100, 1, 10, 100, None, 1, 1, 1, 10, 10, 100, 1, 10, 100, None, 1, 1]
        }

    def data_to_reject(self):
        return {
            "booking_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 18, 19, 20],
            "cost": [1, 10, 10, None, 1, 10, 100, None, 1, 1, 0, 10, 10, 100, 1, 10, 100, None, 1, 1]
        }
