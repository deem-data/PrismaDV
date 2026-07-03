from .abstract_base_case import AbstractBaseCase

# Code and data from https://github.com/patricklowe/Auction_House/blob/main/preproc.py
class GithubWorldofWarcraftAuctionsTimeLeft(AbstractBaseCase):

    def sample_data(self):
        return {
            "auction_id": [653767818, 653768136, 653768140, 653768160, 653768162, 653768196, 653769241, 653769263, 653769285, 653769319, 653769340],
            "quantity": [1, 1, 8, 1, 4, 1, 1, 1, 1, 3, 1],
            "unit_price": [None, None, 1000.0, 4000.0, 30000.0, 4000.0, 3300.0, 33000.0, 14500.0, 6700.0, None],
            "time_left": ["SHORT"] * 11,
            "buyout": [3393900.0, 406200.0, None, None, None, None, None, None, None, None, 60000.0],
            "bid": [None, None, 2289.0, 1180.0, 30183.0, 1711.0, 2290.0, 167738.0, 152576.0, 37118.0, None],
            "context": [0.0, 0.0, None, None, None, None, None, None, None, None, 0.0],
            "collection_year": [2020] * 11,
            "collection_month": [10] * 11,
            "collection_day": [31] * 11,
            "collection_hour": [15] * 11
        }


    def downstream_code(self):
        return """
            auction_df['time_left'] = auction_df['time_left'].replace('SHORT','< 0.5 Hrs').replace('MEDIUM','0.5-2 Hrs').replace('LONG','2-12 Hrs').replace('VERY_LONG','12-48 Hrs')
        """

    def assumption_in_natural_language(self):
        return "The value in the time_left column should be one of the following: 'SHORT','< 0.5 Hrs', 'MEDIUM', '0.5-2 Hrs', 'LONG', '2-12 Hrs', 'VERY_LONG','12-48 Hrs'."

    def target_column(self):
        return "time_left"

    def ground_truth_constraint(self):
        return ".isContainedIn('time_left', ['SHORT','< 0.5 Hrs', 'MEDIUM', '0.5-2 Hrs', 'LONG', '2-12 Hrs', 'VERY_LONG','12-48 Hrs'])"

    def data_to_pass(self):
        return {
            "auction_id": [653767818, 653768136, 653768140, 653768160, 653768162, 653768196, 653769241, 653769263, 653769285, 653769319, 653769340],
            "quantity": [1, 1, 8, 1, 4, 1, 1, 1, 1, 3, 1],
            "unit_price": [None, None, 1000.0, 4000.0, 30000.0, 4000.0, 3300.0, 33000.0, 14500.0, 6700.0, None],
            "time_left": ['SHORT', 'SHORT', '2-12 Hrs', 'SHORT', 'MEDIUM', 'SHORT', 'SHORT', 'SHORT', 'SHORT', 'VERY_LONG', 'SHORT'],
            "buyout": [3393900.0, 406200.0, None, None, None, None, None, None, None, None, 60000.0],
            "bid": [None, None, 2289.0, 1180.0, 30183.0, 1711.0, 2290.0, 167738.0, 152576.0, 37118.0, None],
            "context": [0.0, 0.0, None, None, None, None, None, None, None, None, 0.0],
            "collection_year": [2020] * 11,
            "collection_month": [10] * 11,
            "collection_day": [31] * 11,
            "collection_hour": [15] * 11
        }

    def data_to_reject(self):
        return {
            "auction_id": [653767818, 653768136, 653768140, 653768160, 653768162, 653768196, 653769241, 653769263, 653769285, 653769319, 653769340],
            "quantity": [1, 1, 8, 1, 4, 1, 1, 1, 1, 3, 1],
            "unit_price": [None, None, 1000.0, 4000.0, 30000.0, 4000.0, 3300.0, 33000.0, 14500.0, 6700.0, None],
            "time_left": ['SHORT', 'SHORT', '2-12 Hrs', 'SHORT', 'INFINITELY', 'SHORT', 'SHORT', 'SHORT', 'SHORT', 'VERY_LONG', 'SHORT'],
            "buyout": [3393900.0, 406200.0, None, None, None, None, None, None, None, None, 60000.0],
            "bid": [None, None, 2289.0, 1180.0, 30183.0, 1711.0, 2290.0, 167738.0, 152576.0, 37118.0, None],
            "context": [0.0, 0.0, None, None, None, None, None, None, None, None, 0.0],
            "collection_year": [2020] * 11,
            "collection_month": [10] * 11,
            "collection_day": [31] * 11,
            "collection_hour": [15] * 11
        }