from .abstract_base_case import AbstractBaseCase

class DomainCricketHiddenDependency3(AbstractBaseCase):

    def sample_data(self):
        return {
            "city": [
                "Bangalore", "Chandigarh", "Delhi", "Mumbai", "Kolkata"
            ],
            "date": [
                "2008-04-18", "2008-04-19", "2008-04-19", "2008-04-20", "2008-04-20"
            ],
            "player_of_match": [
                "BB McCullum", "MEK Hussey", "MF Maharoof", "MV Boucher", "DJ Hussey"
            ],
            "venue": [
                "M Chinnaswamy Stadium",
                "Punjab Cricket Association Stadium, Mohali",
                "Feroz Shah Kotla",
                "Wankhede Stadium",
                "Eden Gardens"
            ],
            "team1": [
                "Royal Challengers Bangalore", "Kings XI Punjab", "Delhi Daredevils",
                "Mumbai Indians", "Kolkata Knight Riders"
            ],
            "team2": [
                "Kolkata Knight Riders", "Chennai Super Kings", "Rajasthan Royals",
                "Royal Challengers Bangalore", "Deccan Chargers"
            ],
            "toss_winner": [
                "Royal Challengers Bangalore", "Chennai Super Kings", "Rajasthan Royals",
                "Mumbai Indians", "Deccan Chargers"
            ],
            "toss_decision": [
                "field", "bat", "bat", "bat", "bat"
            ],
            "winner": [
                "Kolkata Knight Riders", "Chennai Super Kings", "Delhi Daredevils",
                "Mumbai Indians", "Kolkata Knight Riders"
            ],
            "result": [
                "runs", "runs", "wickets", "wickets", "wickets"
            ],
            "result_margin": [
                140, 33, 9, 5, 5
            ],
            "target_runs": [
                223, 241, 130, 166, 111
            ],
            "umpire1": [
                "Asad Rauf", "MR Benson", "Aleem Dar", "SJ Davis", "BF Bowden"
            ],
            "umpire2": [
                "RE Koertzen", "SL Shastri", "GA Pratapkumar", "DJ Harper", "K Hariharan"
            ],
        }

    def downstream_code(self):
        return """
            def check_for_collusion(df):
                home_toss_wins = np.sum(df['toss_winner'] == df['team1']) / len(df)
                if home_toss_wins < 0.4 or home_toss_wins > 0.6:
                    raise ValueError("Suspicious distribution of coin toss wins, possible collusion detected.")
        """

    def assumption_in_natural_language(self):
        return "The value in toss_winner should be equal to the value in team1 in roughly half of the cases."

    def target_column(self):
        return "toss_winner"

    def ground_truth_constraint(self):
        return ".satisfies('toss_winner = team1', 'Coin toss should be fair', assertion=lambda x: x >=0.4 and x <= 0.6)"

    def data_to_pass(self):
        return {
            "city": [
                "Bangalore", "Chandigarh", "Delhi", "Mumbai", "Kolkata"
            ],
            "date": [
                "2008-04-18", "2008-04-19", "2008-04-19", "2008-04-20", "2008-04-20"
            ],
            "player_of_match": [
                "BB McCullum", "MEK Hussey", "MF Maharoof", "MV Boucher", "DJ Hussey"
            ],
            "venue": [
                "M Chinnaswamy Stadium",
                "Punjab Cricket Association Stadium, Mohali",
                "Feroz Shah Kotla",
                "Wankhede Stadium",
                "Eden Gardens"
            ],
            "team1": [
                "Royal Challengers Bangalore", "Kings XI Punjab", "Delhi Daredevils",
                "Mumbai Indians", "Kolkata Knight Riders"
            ],
            "team2": [
                "Kolkata Knight Riders", "Chennai Super Kings", "Rajasthan Royals",
                "Royal Challengers Bangalore", "Deccan Chargers"
            ],
            "toss_winner": [
                "Royal Challengers Bangalore", "Kings XI Punjab", "Rajasthan Royals",
                "Mumbai Indians", "Deccan Chargers"
            ],
            "toss_decision": [
                "field", "bat", "bat", "bat", "bat"
            ],
            "winner": [
                "Kolkata Knight Riders", "Chennai Super Kings", "Delhi Daredevils",
                "Mumbai Indians", "Kolkata Knight Riders"
            ],
            "result": [
                "runs", "runs", "wickets", "wickets", "wickets"
            ],
            "result_margin": [
                140, 33, 9, 5, 5
            ],
            "target_runs": [
                223, 241, 130, 166, 111
            ],
            "umpire1": [
                "Asad Rauf", "MR Benson", "Aleem Dar", "SJ Davis", "BF Bowden"
            ],
            "umpire2": [
                "RE Koertzen", "SL Shastri", "GA Pratapkumar", "DJ Harper", "K Hariharan"
            ],
        }

    def data_to_reject(self):
        return {
            "city": [
                "Bangalore", "Chandigarh", "Delhi", "Mumbai", "Kolkata"
            ],
            "date": [
                "2008-04-18", "2008-04-19", "2008-04-19", "2008-04-20", "2008-04-20"
            ],
            "player_of_match": [
                "BB McCullum", "MEK Hussey", "MF Maharoof", "MV Boucher", "DJ Hussey"
            ],
            "venue": [
                "M Chinnaswamy Stadium",
                "Punjab Cricket Association Stadium, Mohali",
                "Feroz Shah Kotla",
                "Wankhede Stadium",
                "Eden Gardens"
            ],
            "team1": [
                "Royal Challengers Bangalore", "Kings XI Punjab", "Delhi Daredevils",
                "Mumbai Indians", "Kolkata Knight Riders"
            ],
            "team2": [
                "Kolkata Knight Riders", "Chennai Super Kings", "Rajasthan Royals",
                "Royal Challengers Bangalore", "Deccan Chargers"
            ],
            "toss_winner": [
                "Kolkata Knight Riders", "Chennai Super Kings", "Rajasthan Royals",
                "Royal Challengers Bangalore", "Deccan Chargers"
            ],
            "toss_decision": [
                "field", "bat", "bat", "bat", "bat"
            ],
            "winner": [
                "Kolkata Knight Riders", "Chennai Super Kings", "Delhi Daredevils",
                "Mumbai Indians", "Kolkata Knight Riders"
            ],
            "result": [
                "runs", "runs", "wickets", "wickets", "wickets"
            ],
            "result_margin": [
                140, 33, 9, 5, 5
            ],
            "target_runs": [
                223, 241, 130, 166, 111
            ],
            "umpire1": [
                "Asad Rauf", "MR Benson", "Aleem Dar", "SJ Davis", "BF Bowden"
            ],
            "umpire2": [
                "RE Koertzen", "SL Shastri", "GA Pratapkumar", "DJ Harper", "K Hariharan"
            ],
        }