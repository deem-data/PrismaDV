from .abstract_base_case import AbstractBaseCase


class KnowledgeInSQLColumnCondition(AbstractBaseCase):

    def sample_data(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "email": ["user1@mail.com", "user2@mail.com", None, "user4@mail.com"],
            "cost": [1, 10, 10, 100],
            "status": ["UNDER_REVIEW", "COMPLETED", "CANCELLED", "CANCELLED"],
        }

    def downstream_code(self):
        return """
        import duckdb
        conn = duckdb.connect()
        
        result = duckdb.execute('''
        SELECT email FROM df WHERE status = 'COMPLETED'
        ''').fetchall()
        
        for row in result:
            send_confirmation_mail(row[0])
        """

    def assumption_in_natural_language(self):
        return "There should be a valid email for each booking with status 'COMPLETED'."

    def target_column(self):
        return "email"

    def ground_truth_constraint(self):
        return ".isComplete('email').where('status = \"COMPLETED\"')"

    def data_to_pass(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "email": ["user1@mail.com", "user2@mail.com", "user3@mail.com", None],
            "cost": [1, 10, 10, 100],
            "status": ["COMPLETED", "COMPLETED", "CANCELLED", "UNDER_REVIEW"],
        }

    def data_to_reject(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "email": ["user1@mail.com", "user2@mail.com", None, "user4@mail.com"],
            "cost": [1, 10, 10, 100],
            "status": ["COMPLETED", "COMPLETED", "COMPLETED", "CANCELLED"],
        }
