from .abstract_base_case import AbstractBaseCase

# Code and data from https://github.com/davide97l/WoW-dataset-analysis
class GithubIPLMatchWinnerRules2(AbstractBaseCase):

    def sample_data(self):
        return {
            "innings": [1]*5,
            "overs": [0, 0, 0, 0, 0],
            "ballnumber": [1, 2, 3, 4, 5],
            "batter": ["YBK Jaiswal", "YBK Jaiswal", "JC Buttler", "YBK Jaiswal", "YBK Jaiswal"],
            "bowler": ["Mohammed Shami"]*5,
            "non-striker": ["JC Buttler", "JC Buttler", "YBK Jaiswal", "JC Buttler", "JC Buttler"],
            "extra_type": ["NA", "legbyes", "NA", "NA", "NA"],
            "batsman_run": [0, 0, 1, 0, 0],
            "extras_run": [0, 1, 0, 0, 0],
            "total_run": [0, 1, 1, 0, 0],
            "non_boundary": [0]*5,
            "isWicketDelivery": [0]*5,
            "player_out": ["NA"]*5,
            "kind": ["NA"]*5,
            "fielders_involved": ["NA"]*5,
            "BattingTeam": ["Rajasthan Royals"]*5
        }


    def downstream_code(self):
        return """
            delivery_df['balls_left'] = 120 - (delivery_df['overs']*6 + delivery_df['ballnumber'])
            delivery_df['wickets'] = 10 - wickets        
        """

    def assumption_in_natural_language(self):
        return "The value in the ballnumber column must be between 0 and 6."

    def target_column(self):
        return "ballnumber"

    def ground_truth_constraint(self):
        return "satisfies('ballnumber >= 0 AND ballnumber <= 6', 'Overs range')"

    def data_to_pass(self):
        return {
            "innings": [1]*5,
            "overs": [0, 0, 0, 0, 0],
            "ballnumber": [1, 2, 3, 4, 6],
            "batter": ["YBK Jaiswal", "YBK Jaiswal", "JC Buttler", "YBK Jaiswal", "YBK Jaiswal"],
            "bowler": ["Mohammed Shami"]*5,
            "non-striker": ["JC Buttler", "JC Buttler", "YBK Jaiswal", "JC Buttler", "JC Buttler"],
            "extra_type": ["NA", "legbyes", "NA", "NA", "NA"],
            "batsman_run": [0, 0, 1, 0, 0],
            "extras_run": [0, 1, 0, 0, 0],
            "total_run": [0, 1, 1, 0, 0],
            "non_boundary": [0]*5,
            "isWicketDelivery": [0]*5,
            "player_out": ["NA"]*5,
            "kind": ["NA"]*5,
            "fielders_involved": ["NA"]*5,
            "BattingTeam": ["Rajasthan Royals"]*5
        }

    def data_to_reject(self):
        return {
            "innings": [1]*5,
            "overs": [0, 0, 0, 0, 0],
            "ballnumber": [1, 2, 3, 4, 25],
            "batter": ["YBK Jaiswal", "YBK Jaiswal", "JC Buttler", "YBK Jaiswal", "YBK Jaiswal"],
            "bowler": ["Mohammed Shami"]*5,
            "non-striker": ["JC Buttler", "JC Buttler", "YBK Jaiswal", "JC Buttler", "JC Buttler"],
            "extra_type": ["NA", "legbyes", "NA", "NA", "NA"],
            "batsman_run": [0, 0, 1, 0, 0],
            "extras_run": [0, 1, 0, 0, 0],
            "total_run": [0, 1, 1, 0, 0],
            "non_boundary": [0]*5,
            "isWicketDelivery": [0]*5,
            "player_out": ["NA"]*5,
            "kind": ["NA"]*5,
            "fielders_involved": ["NA"]*5,
            "BattingTeam": ["Rajasthan Royals"]*5
        }