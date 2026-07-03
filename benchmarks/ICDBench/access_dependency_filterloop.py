from .abstract_base_case import AbstractBaseCase


class AccessDependencyFilterLoop(AbstractBaseCase):

    def sample_data(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "email": ["user1@mail.com", "user2@mail.com", None, "user4@mail.com"],
            "cost": [1, 10, 10, 100],
            "status": ["UNDER_REVIEW", "COMPLETED", "CANCELLED", "CANCELLED"],
        }

    def downstream_code(self):
        return """
        df = df.dropna(subset=["status"])
        completed_bookings = df[df['status'] == "COMPLETED"]
        for row in completed_bookings.iterrows():
            send_email(row['email'], "Your booking is completed.")
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
