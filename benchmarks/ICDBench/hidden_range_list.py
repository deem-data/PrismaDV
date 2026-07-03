from .abstract_base_case import AbstractBaseCase


class HiddenRangeList(AbstractBaseCase):

    def sample_data(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "cost": [1, 10, 10, 100],
            "status": [0, 0, 1, 1],
        }

    def downstream_code(self):
        return """
        action_per_status = ["send_money", "rebook", "log", "notify_legal"]
        
        for row in df.iterrows():
            if not pd.isna(row['status']):
                action_to_take = action_per_status[row['status']]
                schedule_next_action(row['booking_id'], action_to_take)
        """

    def assumption_in_natural_language(self):
        return "All non-null values in the status column should be have one of the following values: 0, 1, 2, 3."

    def target_column(self):
        return "status"

    def ground_truth_constraint(self):
        return ".isContainedIn('status', [0, 1, 2, 3])"

    def data_to_pass(self):
        return {
            "booking_id": [5, 6, 7, 8],
            "cost": [10, 1000, 10, 100],
            "status": [0, 2, 1, 3],
        }

    def data_to_reject(self):
        return {
            "booking_id": [5, 6, 7, 8],
            "cost": [10, 0, 10, 100],
            "status": [0, 2, 1, 4],
        }
