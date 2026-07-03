from .abstract_base_case import AbstractBaseCase


class SklearnLabelComplete(AbstractBaseCase):

    def sample_data(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "cost": [1, 10, 10, 100],
            "location": ["US", "EU", "US", "EU"],
            "status": [0, 1, 1, 0],
        }

    def downstream_code(self):
        return """
        import math
        import numpy as np
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.linear_model import LogisticRegression
        known_locations = ['US', 'EU', 'AS', 'AFR', 'AUS']
        locations = OneHotEncoder(categories=known_locations, sparse_output=False, handle_unknown='error') \
            .transform(df[['location']])
        X = np.column_stack((locations, df['cost'].values.reshape(-1, 1)))
        y = df['status']
        model = LogisticRegression().fit(X, y)
        deploy_model(model)            
        """

    def assumption_in_natural_language(self):
        return "The status column must be complete."

    def target_column(self):
        return "status"

    def ground_truth_constraint(self):
        return ".isComplete('status')"

    def data_to_pass(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "cost": [1, -10, 10, -100],
            "location": ["US", "AS", "AFR", "AUS"],
            "status": [0, 1, 1, 1],
        }

    def data_to_reject(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "cost": [1, 10, None, 100],
            "location": ["A", "FRANCE", "GERMANY", "C"],
            "status": [0, None, 1, None],
        }
