from .abstract_base_case import AbstractBaseCase


class SklearnLogregNanWithCondition(AbstractBaseCase):

    def sample_data(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "cost": [1, 10, 10, 100],
            "location": ["A", "B", "A", "C"],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }

    def downstream_code(self):
        return """
        import math
        import numpy as np
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.linear_model import LogisticRegression
        df = df[df['location'] == 'A']
        locations = OneHotEncoder(sparse_output=False).fit_transform(df['location'])
        X = np.column_stack((locations, df['cost'].values.reshape(-1, 1)))
        y = df['status']=='COMPLETED'
        model = LogisticRegression().fit(X, y)
        deploy_model(model)            
        """

    def assumption_in_natural_language(self):
        return "The cost column should be complete for rows where location is 'A'."

    def target_column(self):
        return "cost"

    def ground_truth_constraint(self):
        return ".isComplete('cost').where('location = \"A\"')"

    def data_to_pass(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "cost": [1, None, 10, -100],
            "location": ["A", "B", "A", "C"],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }

    def data_to_reject(self):
        return {
            "booking_id": [1, 2, 3, 4],
            "cost": [1, 10, None, 100],
            "location": ["A", "B", "A", "C"],
            "status": ["COMPLETED", "IN_PROGRESS", "COMPLETED", "CANCELLED"],
        }
