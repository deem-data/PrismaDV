from .abstract_base_case import AbstractBaseCase

# Code and data from https://github.com/SimranS22/IPL-MATCH-WINNER-PREDICTOR/blob/main/notebooks/IPL_LR.ipynb
class GithubIPLMatchWinnerCity(AbstractBaseCase):

    def sample_data(self):
        return {
            "City": ["Ahmedabad", "Ahmedabad", "Kolkata", "Kolkata", "Mumbai", "Mumbai", "Mumbai", "Mumbai", "Navi Mumbai", "Mumbai"],
            "Date": ["2022-05-29", "2022-05-27", "2022-05-25", "2022-05-24", "2022-05-22", "2022-05-21", "2022-05-20", "2022-05-19", "2022-05-18", "2022-05-17"],
            "MatchNumber": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1", "70", "69", "68", "67", "66", "65"],
            "Team1": ["Rajasthan Royals", "Royal Challengers Bangalore", "Royal Challengers Bangalore", "Rajasthan Royals",
                      "Sunrisers Hyderabad", "Delhi Capitals", "Chennai Super Kings", "Gujarat Titans", "Lucknow Super Giants", "Sunrisers Hyderabad"],
            "Team2": ["Gujarat Titans", "Rajasthan Royals", "Lucknow Super Giants", "Gujarat Titans",
                      "Punjab Kings", "Mumbai Indians", "Rajasthan Royals", "Royal Challengers Bangalore", "Kolkata Knight Riders", "Mumbai Indians"],
            "TossWinner": ["Rajasthan Royals", "Rajasthan Royals", "Lucknow Super Giants", "Gujarat Titans",
                           "Sunrisers Hyderabad", "Mumbai Indians", "Chennai Super Kings", "Gujarat Titans", "Lucknow Super Giants", "Mumbai Indians"],
            "TossDecision": ["bat", "field", "field", "field", "bat", "field", "bat", "bat", "bat", "field"],
            "SuperOver": ["N"] * 10,
            "WinningTeam": ["Gujarat Titans", "Rajasthan Royals", "Royal Challengers Bangalore", "Gujarat Titans",
                            "Punjab Kings", "Mumbai Indians", "Rajasthan Royals", "Royal Challengers Bangalore", "Lucknow Super Giants", "Sunrisers Hyderabad"],
            "WonBy": ["Wickets", "Wickets", "Runs", "Wickets", "Wickets", "Wickets", "Wickets", "Wickets", "Runs", "Runs"],
        }


    def downstream_code(self):
        return """
            delivery_df['Team1'] = delivery_df['Team1'].str.replace('Delhi Daredevils','Delhi Capitals')
            delivery_df['Team2'] = delivery_df['Team2'].str.replace('Delhi Daredevils','Delhi Capitals')
            delivery_df['WinningTeam'] = delivery_df['WinningTeam'].str.replace('Delhi Daredevils','Delhi Capitals')
            delivery_df['BattingTeam'] = delivery_df['BattingTeam'].str.replace('Delhi Daredevils','Delhi Capitals')
            
            delivery_df['Team1'] = delivery_df['Team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
            delivery_df['Team2'] = delivery_df['Team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
            delivery_df['WinningTeam'] = delivery_df['WinningTeam'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
            delivery_df['BattingTeam'] = delivery_df['BattingTeam'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
            
            delivery_df['Team1'] = delivery_df['Team1'].str.replace('Gujarat Lions','Gujarat Titans')
            delivery_df['Team2'] = delivery_df['Team2'].str.replace('Gujarat Lions','Gujarat Titans')
            delivery_df['WinningTeam'] = delivery_df['WinningTeam'].str.replace('Gujarat Lions','Gujarat Titans')
            delivery_df['BattingTeam'] = delivery_df['BattingTeam'].str.replace('Gujarat Lions','Gujarat Titans')
            
            delivery_df['Team1'] = delivery_df['Team1'].str.replace('Kings XI Punjab','Punjab Kings')
            delivery_df['Team2'] = delivery_df['Team2'].str.replace('Kings XI Punjab','Punjab Kings')
            delivery_df['WinningTeam'] = delivery_df['WinningTeam'].str.replace('Kings XI Punjab','Punjab Kings')
            delivery_df['BattingTeam'] = delivery_df['BattingTeam'].str.replace('Kings XI Punjab','Punjab Kings')
            
            
            delivery_df['City'] = delivery_df['City'].str.replace('Bangalore','Bengaluru')
            delivery_df['City'] = delivery_df['City'].str.replace('Dharamsala','Dharamshala')        
        """

    def assumption_in_natural_language(self):
        return "The values in the City column must be one of the following: 'Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Bangalore','Bengaluru', 'Dharamsala','Dharamshala'."

    def target_column(self):
        return "City"

    def ground_truth_constraint(self):
        return "isContainedIn('City', ['Ahmedabad', 'Kolkata', 'Mumbai', 'Navi Mumbai', 'Bangalore','Bengaluru', 'Dharamsala','Dharamshala'])"


    def data_to_pass(self):
        return {
            "City": ["Dharamshala", "Ahmedabad", "Kolkata", "Kolkata", "Mumbai", "Mumbai", "Mumbai", "Mumbai", "Navi Mumbai", "Mumbai"],
            "Date": ["2022-05-29", "2022-05-27", "2022-05-25", "2022-05-24", "2022-05-22", "2022-05-21", "2022-05-20", "2022-05-19", "2022-05-18", "2022-05-17"],
            "MatchNumber": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1", "70", "69", "68", "67", "66", "65"],
            "Team1": ["Rajasthan Royals", "Royal Challengers Bangalore", "Royal Challengers Bangalore", "Rajasthan Royals",
                      "Sunrisers Hyderabad", "Delhi Capitals", "Chennai Super Kings", "Gujarat Titans", "Lucknow Super Giants", "Sunrisers Hyderabad"],
            "Team2": ["Gujarat Titans", "Rajasthan Royals", "Lucknow Super Giants", "Gujarat Titans",
                      "Punjab Kings", "Mumbai Indians", "Rajasthan Royals", "Royal Challengers Bangalore", "Kolkata Knight Riders", "Mumbai Indians"],
            "TossWinner": ["Rajasthan Royals", "Rajasthan Royals", "Lucknow Super Giants", "Gujarat Titans",
                           "Sunrisers Hyderabad", "Mumbai Indians", "Chennai Super Kings", "Gujarat Titans", "Lucknow Super Giants", "Mumbai Indians"],
            "TossDecision": ["bat", "field", "field", "field", "bat", "field", "bat", "bat", "bat", "field"],
            "SuperOver": ["N"] * 10,
            "WinningTeam": ["Gujarat Titans", "Rajasthan Royals", "Royal Challengers Bangalore", "Gujarat Titans",
                            "Punjab Kings", "Mumbai Indians", "Rajasthan Royals", "Royal Challengers Bangalore", "Lucknow Super Giants", "Sunrisers Hyderabad"],
            "WonBy": ["Wickets", "Wickets", "Runs", "Wickets", "Wickets", "Wickets", "Wickets", "Wickets", "Runs", "Runs"],
        }

    def data_to_reject(self):
        return {
            "City": ["Berlin", "Ahmedabad", "Kolkata", "Kolkata", "Mumbai", "Mumbai", "Mumbai", "Mumbai", "Navi Mumbai", "Mumbai"],
            "Date": ["2022-05-29", "2022-05-27", "2022-05-25", "2022-05-24", "2022-05-22", "2022-05-21", "2022-05-20", "2022-05-19", "2022-05-18", "2022-05-17"],
            "MatchNumber": ["Final", "Qualifier 2", "Eliminator", "Qualifier 1", "70", "69", "68", "67", "66", "65"],
            "Team1": ["Rajasthan Royals", "Royal Challengers Bangalore", "Royal Challengers Bangalore", "Rajasthan Royals",
                      "Sunrisers Hyderabad", "Delhi Capitals", "Chennai Super Kings", "Gujarat Titans", "Lucknow Super Giants", "Sunrisers Hyderabad"],
            "Team2": ["Gujarat Titans", "Rajasthan Royals", "Lucknow Super Giants", "Gujarat Titans",
                      "Punjab Kings", "Mumbai Indians", "Rajasthan Royals", "Royal Challengers Bangalore", "Kolkata Knight Riders", "Mumbai Indians"],
            "TossWinner": ["Rajasthan Royals", "Rajasthan Royals", "Lucknow Super Giants", "Gujarat Titans",
                           "Sunrisers Hyderabad", "Mumbai Indians", "Chennai Super Kings", "Gujarat Titans", "Lucknow Super Giants", "Mumbai Indians"],
            "TossDecision": ["bat", "field", "field", "field", "bat", "field", "bat", "bat", "bat", "field"],
            "SuperOver": ["N"] * 10,
            "WinningTeam": ["Gujarat Titans", "Rajasthan Royals", "Royal Challengers Bangalore", "Gujarat Titans",
                            "Punjab Kings", "Mumbai Indians", "Rajasthan Royals", "Royal Challengers Bangalore", "Lucknow Super Giants", "Sunrisers Hyderabad"],
            "WonBy": ["Wickets", "Wickets", "Runs", "Wickets", "Wickets", "Wickets", "Wickets", "Wickets", "Runs", "Runs"],
        }
