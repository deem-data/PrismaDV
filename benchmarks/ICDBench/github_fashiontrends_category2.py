from .abstract_base_case import AbstractBaseCase

# Code and data from https://github.com/macedoti13/Future-Fashion-Trends-Forecasting/blob/main/sentiment_analysis_model.py
class GithubFashionTrendsCategory2(AbstractBaseCase):

    def sample_data(self):
        return {
            "comments": [
                "I have a proposal for u, send me a DM please!",
                "Woaaaah",
                ":smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes:",
                ":red_heart:️:red_heart:️:red_heart:️",
                ":smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes: so glad you made it! @kilianparis",
                ":fire::fire::fire:",
                "This is perfection :smiling_face_with_heart-eyes:",
                "The best ever :fire:",
                "You're so gorgeous :smiling_face_with_heart-eyes:",
                "Stunning as usual :red_heart:️",
                "Hey Tina I just want to make sure you got my message :)",
                "All the colors! these are stunning!",
                "How did moccona get your home address though ?",
                "So pretty :smiling_face_with_heart-eyes:",
                "Just caught up on this. Beautiful words :red_heart:️",
                "@cristinadalton22",
                "love this monochrome look :fire::fire::fire:",
                ":smiling_face_with_heart-eyes:",
                "Mary, this is incredible :smiling_face_with_heart-eyes:",
                "The drama:fire:",
                "Love this look :smiling_face_with_heart-eyes:"
            ],
            "category": [
                0.0, 0.0, 1.0, 1.0, 0.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 0.0, 1.0,
                1.0
            ]
        }

    def downstream_code(self):
        return '''
            def split_data(df):
                """Split the data into training and testing sets.
            
                Args:
                df (pd.DataFrame): The data to split.
            
            Returns:
            list: The split data [X_train, X_test, y_train, y_test].
            """
            df2 = df[df['category'].notnull()]
            X = df2.comments
            y = df2.category
            return train_test_split(X, y, test_size=0.2)        
    '''

    def assumption_in_natural_language(self):
        return "The category column must have at least two distinct non-null values."

    def target_column(self):
        return "category"

    def ground_truth_constraint(self):
        return "hasNumberOfDistinctValues('category', lambda x: x >= 2)"

    def data_to_pass(self):
        return {
            "comments": [
                "I have a proposal for u, send me a DM please!",
                "Woaaaah",
                ":smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes:",
                ":red_heart:️:red_heart:️:red_heart:️",
                ":smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes: so glad you made it! @kilianparis",
                ":fire::fire::fire:",
                "This is perfection :smiling_face_with_heart-eyes:",
                "The best ever :fire:",
                "You're so gorgeous :smiling_face_with_heart-eyes:",
                "Stunning as usual :red_heart:️",
                "Hey Tina I just want to make sure you got my message :)",
                "All the colors! these are stunning!",
                "How did moccona get your home address though ?",
                "So pretty :smiling_face_with_heart-eyes:",
                "Just caught up on this. Beautiful words :red_heart:️",
                "@cristinadalton22",
                "love this monochrome look :fire::fire::fire:",
                ":smiling_face_with_heart-eyes:",
                "Mary, this is incredible :smiling_face_with_heart-eyes:",
                "The drama:fire:",
                "Love this look :smiling_face_with_heart-eyes:"
            ],
            "category": [
                0.0, None, 1.0, 1.0, 0.0,
                1.0, 1.0, 1.0, 1.0, 1.0,
                0.0, 1.0, 0.0, 1.0, 0.0,
                0.0, None, 1.0, 0.0, 1.0,
                1.0
            ]
        }

    def data_to_reject(self):
        return {
            "comments": [
                "I have a proposal for u, send me a DM please!",
                "Woaaaah",
                ":smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes:",
                ":red_heart:️:red_heart:️:red_heart:️",
                ":smiling_face_with_heart-eyes::smiling_face_with_heart-eyes::smiling_face_with_heart-eyes: so glad you made it! @kilianparis",
                ":fire::fire::fire:",
                "This is perfection :smiling_face_with_heart-eyes:",
                "The best ever :fire:",
                "You're so gorgeous :smiling_face_with_heart-eyes:",
                "Stunning as usual :red_heart:️",
                "Hey Tina I just want to make sure you got my message :)",
                "All the colors! these are stunning!",
                "How did moccona get your home address though ?",
                "So pretty :smiling_face_with_heart-eyes:",
                "Just caught up on this. Beautiful words :red_heart:️",
                "@cristinadalton22",
                "love this monochrome look :fire::fire::fire:",
                ":smiling_face_with_heart-eyes:",
                "Mary, this is incredible :smiling_face_with_heart-eyes:",
                "The drama:fire:",
                "Love this look :smiling_face_with_heart-eyes:"
            ],
            "category": [
                None, None, None, None, None,
                None, None, 1.0, None, None,
                1.0, None, 1.0, None, None,
                None, None, None, None, None,
                None
            ]
        }