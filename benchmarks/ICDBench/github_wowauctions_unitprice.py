from .abstract_base_case import AbstractBaseCase

# Code and data from https://github.com/patricklowe/Auction_House/blob/main/preproc.py
class GithubWorldofWarcraftAuctionsUnitPrice(AbstractBaseCase):

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
        # Unit_price are for stackable items, and buyout price is for non-stackable items. Let's combine into one cost source
        auction_df['cost'] = (auction_df['unit_price'].fillna(0) + auction_df['buyout'].fillna(0)).astype(str)
        
        # Converting cost into gold (100 silver), silver (100 copper) and copper pieces
        auction_df['cost_gold'] = auction_df['cost'].astype(str).str[:-6]
        auction_df['cost_silver'] = auction_df['cost'].astype(str).str[-6:-4]       
        auction_df['cost_copper'] = auction_df['cost'].astype(str).str[-4:-2] 
        """

    def assumption_in_natural_language(self):
        return "Either unit_price or buyout must be greater than 0.0, but not both at the same time."

    def target_column(self):
        return "unit_price"

    def ground_truth_constraint(self):
        return ".satisfies('(unit_price > 0.0 AND buyout IS NULL) OR (unit_price IS NULL AND buyout > 0.0) OR (unit_price > 0.0 AND buyout > 0.0)', 'Valid auction price')"

    def data_to_pass(self):
        return {
            "auction_id": [653767818, 653768136, 653768140, 653768160, 653768162, 653768196, 653769241, 653769263, 653769285, 653769319, 653769340],
            "quantity": [1, 1, 8, 1, 4, 1, 1, 1, 1, 3, 1],
            "unit_price": [100.0, None, 1000.0, 4000.0, 30000.0, 4000.0, 3300.0, 33000.0, 14500.0, 6700.0, None],
            "time_left": ["SHORT"] * 11,
            "buyout": [None, 406200.0, None, None, None, None, None, None, None, None, 60000.0],
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
            "time_left": ["SHORT"] * 11,
            "buyout": [None, 406200.0, None, None, None, None, None, None, None, None, 60000.0],
            "bid": [None, None, 2289.0, 1180.0, 30183.0, 1711.0, 2290.0, 167738.0, 152576.0, 37118.0, None],
            "context": [0.0, 0.0, None, None, None, None, None, None, None, None, 0.0],
            "collection_year": [2020] * 11,
            "collection_month": [10] * 11,
            "collection_day": [31] * 11,
            "collection_hour": [15] * 11
        }