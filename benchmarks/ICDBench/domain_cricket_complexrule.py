from .abstract_base_case import AbstractBaseCase

class DomainCricketComplexRule(AbstractBaseCase):

    def sample_data(self):
        return {
            "city": [
                "Bangalore", "Chandigarh", "Delhi", "Mumbai", "Kolkata",
                "Jaipur", "Hyderabad", "Chennai", "Hyderabad", "Chandigarh",
                "Bangalore", "Chennai", "Mumbai", "Chandigarh", "Bangalore"
            ],
            "date": [
                "2008-04-18", "2008-04-19", "2008-04-19", "2008-04-20", "2008-04-20",
                "2008-04-21", "2008-04-22", "2008-04-23", "2008-04-24", "2008-04-25",
                "2008-04-26", "2008-04-26", "2008-04-27", "2008-04-27", "2008-04-28"
            ],
            "player_of_match": [
                "BB McCullum", "MEK Hussey", "MF Maharoof", "MV Boucher", "DJ Hussey",
                "SR Watson", "V Sehwag", "ML Hayden", "YK Pathan", "KC Sangakkara",
                "SR Watson", "JDP Oram", "AC Gilchrist", "SM Katich", "MS Dhoni"
            ],
            "venue": [
                "M Chinnaswamy Stadium", "Punjab Cricket Association Stadium, Mohali", "Feroz Shah Kotla",
                "Wankhede Stadium", "Eden Gardens", "Sawai Mansingh Stadium",
                "Rajiv Gandhi International Stadium, Uppal", "MA Chidambaram Stadium, Chepauk",
                "Rajiv Gandhi International Stadium, Uppal", "Punjab Cricket Association Stadium, Mohali",
                "M Chinnaswamy Stadium", "MA Chidambaram Stadium, Chepauk",
                "Dr DY Patil Sports Academy", "Punjab Cricket Association Stadium, Mohali",
                "M Chinnaswamy Stadium"
            ],
            "team1": [
                "Royal Challengers Bangalore", "Kings XI Punjab", "Delhi Daredevils", "Mumbai Indians",
                "Kolkata Knight Riders", "Rajasthan Royals", "Deccan Chargers", "Chennai Super Kings",
                "Deccan Chargers", "Kings XI Punjab", "Royal Challengers Bangalore",
                "Chennai Super Kings", "Mumbai Indians", "Kings XI Punjab", "Royal Challengers Bangalore"
            ],
            "team2": [
                "Kolkata Knight Riders", "Chennai Super Kings", "Rajasthan Royals", "Royal Challengers Bangalore",
                "Deccan Chargers", "Kings XI Punjab", "Delhi Daredevils", "Mumbai Indians",
                "Rajasthan Royals", "Mumbai Indians", "Rajasthan Royals", "Kolkata Knight Riders",
                "Deccan Chargers", "Delhi Daredevils", "Chennai Super Kings"
            ],
            "toss_winner": [
                "Royal Challengers Bangalore", "Chennai Super Kings", "Rajasthan Royals", "Mumbai Indians",
                "Deccan Chargers", "Kings XI Punjab", "Deccan Chargers", "Mumbai Indians",
                "Rajasthan Royals", "Mumbai Indians", "Rajasthan Royals", "Kolkata Knight Riders",
                "Mumbai Indians", "Kings XI Punjab", "Chennai Super Kings"
            ],
            "toss_decision": [
                "field", "bat", "bat", "bat", "bat", "bat", "bat", "field", "field", "field",
                "field", "bat", "field", "bat", "bat"
            ],
            "winner": [
                "Kolkata Knight Riders", "Chennai Super Kings", "Delhi Daredevils", "Mumbai Indians",
                "Kolkata Knight Riders", "Kings XI Punjab", "Delhi Daredevils", "Chennai Super Kings",
                "Rajasthan Royals", "Kings XI Punjab", "Rajasthan Royals", "Kolkata Knight Riders",
                "Deccan Chargers", "Delhi Daredevils", "Chennai Super Kings"
            ],
            "result": [
                "runs", "runs", "wickets", "wickets", "wickets",
                "wickets", "wickets", "runs", "wickets", "runs",
                "wickets", "wickets", "wickets", "wickets", "runs"
            ],
            "result_margin": [
                140, 33, 9, 5, 5, 6, 9, 6, 3, 66, 7, 9, 10, 4, 13
            ],
            "target_runs": [
                223, 241, 130, 166, 111, 167, 143, 209, 215, 183, 136, 148, 155, 159, 179
            ],
            "target_overs": [
                20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20
            ],
            "umpire1": [
                "Asad Rauf", "MR Benson", "Aleem Dar", "SJ Davis", "BF Bowden",
                "Aleem Dar", "IL Howell", "DJ Harper", "Asad Rauf", "Aleem Dar",
                "MR Benson", "BF Bowden", "Asad Rauf", "RE Koertzen", "BR Doctrove"
            ],
            "umpire2": [
                "RE Koertzen", "SL Shastri", "GA Pratapkumar", "DJ Harper", "K Hariharan",
                "RB Tiffin", "AM Saheba", "GA Pratapkumar", "MR Benson", "AM Saheba",
                "IL Howell", "AV Jayaprakash", "SL Shastri", "I Shivram", "RB Tiffin"
            ],
        }

    def downstream_code(self):
        return '''
        for row in df.iterrows():
            winner = row['winner']
            toss_winner = row['toss_winner']
            toss_decision = row['toss_decision']
            result = row['result']
            
            if toss_decision == 'bat' and winner == toss_winner:
                assert result == 'runs'
            if toss_decision == 'field' and winner != toss_winner:
                assert result == 'runs'                
            if toss_decision == 'field' and winner == toss_winner:
                assert result == 'wickets'
            if toss_decision == 'bat' and winner != toss_winner:
                assert result == 'wickets'                
        '''

    def assumption_in_natural_language(self):
        return "If the team that batted first wins, the result can only be runs, and if the team that fielded first wins, the result can only be wickets. To identify who batted first, you need to check the toss_winner and toss_decision."

    def target_column(self):
        return "result"

    def ground_truth_constraint(self):
        return ".satisfies(\"(toss_decision = 'bat' AND winner = toss_winner AND result = 'runs') OR (toss_decision = 'field' AND winner != toss_winner AND result = 'runs') OR (toss_decision = 'field' AND winner = toss_winner AND result = 'wickets') OR (toss_decision = 'bat' AND winner != toss_winner AND result = 'wickets')\", 'Cricket match result should be consistent with toss decision and winner')"

    def data_to_pass(self):
        return {
            "city": [
                "Bangalore", "Chandigarh", "Delhi", "Mumbai", "Kolkata",
                "Jaipur", "Hyderabad", "Chennai", "Hyderabad", "Chandigarh",
                "Bangalore", "Chennai", "Mumbai", "Chandigarh", "Bangalore"
            ],
            "date": [
                "2008-04-18", "2008-04-19", "2008-04-19", "2008-04-20", "2008-04-20",
                "2008-04-21", "2008-04-22", "2008-04-23", "2008-04-24", "2008-04-25",
                "2008-04-26", "2008-04-26", "2008-04-27", "2008-04-27", "2008-04-28"
            ],
            "player_of_match": [
                "BB McCullum", "MEK Hussey", "MF Maharoof", "MV Boucher", "DJ Hussey",
                "SR Watson", "V Sehwag", "ML Hayden", "YK Pathan", "KC Sangakkara",
                "SR Watson", "JDP Oram", "AC Gilchrist", "SM Katich", "MS Dhoni"
            ],
            "venue": [
                "M Chinnaswamy Stadium", "Punjab Cricket Association Stadium, Mohali", "Feroz Shah Kotla",
                "Wankhede Stadium", "Eden Gardens", "Sawai Mansingh Stadium",
                "Rajiv Gandhi International Stadium, Uppal", "MA Chidambaram Stadium, Chepauk",
                "Rajiv Gandhi International Stadium, Uppal", "Punjab Cricket Association Stadium, Mohali",
                "M Chinnaswamy Stadium", "MA Chidambaram Stadium, Chepauk",
                "Dr DY Patil Sports Academy", "Punjab Cricket Association Stadium, Mohali",
                "M Chinnaswamy Stadium"
            ],
            "team1": [
                "Royal Challengers Bangalore", "Kings XI Punjab", "Delhi Daredevils", "Mumbai Indians",
                "Kolkata Knight Riders", "Rajasthan Royals", "Deccan Chargers", "Chennai Super Kings",
                "Deccan Chargers", "Kings XI Punjab", "Royal Challengers Bangalore",
                "Chennai Super Kings", "Mumbai Indians", "Kings XI Punjab", "Royal Challengers Bangalore"
            ],
            "team2": [
                "Kolkata Knight Riders", "Chennai Super Kings", "Rajasthan Royals", "Royal Challengers Bangalore",
                "Deccan Chargers", "Kings XI Punjab", "Delhi Daredevils", "Mumbai Indians",
                "Rajasthan Royals", "Mumbai Indians", "Rajasthan Royals", "Kolkata Knight Riders",
                "Deccan Chargers", "Delhi Daredevils", "Chennai Super Kings"
            ],
            "toss_winner": [
                "Royal Challengers Bangalore", "Chennai Super Kings", "Rajasthan Royals", "Mumbai Indians",
                "Deccan Chargers", "Kings XI Punjab", "Deccan Chargers", "Mumbai Indians",
                "Rajasthan Royals", "Mumbai Indians", "Rajasthan Royals", "Kolkata Knight Riders",
                "Mumbai Indians", "Kings XI Punjab", "Chennai Super Kings"
            ],
            "toss_decision": [
                "bat", "bat", "bat", "bat", "bat", "bat", "bat", "field", "field", "field",
                "field", "bat", "field", "bat", "bat"
            ],
            "winner": [
                "Kolkata Knight Riders", "Chennai Super Kings", "Delhi Daredevils", "Mumbai Indians",
                "Kolkata Knight Riders", "Kings XI Punjab", "Delhi Daredevils", "Chennai Super Kings",
                "Rajasthan Royals", "Kings XI Punjab", "Rajasthan Royals", "Kolkata Knight Riders",
                "Deccan Chargers", "Delhi Daredevils", "Chennai Super Kings"
            ],
            "result": [
                "wickets", "runs", "wickets", "wickets", "wickets",
                "wickets", "wickets", "runs", "wickets", "runs",
                "wickets", "wickets", "wickets", "wickets", "runs"
            ],
            "result_margin": [
                9, 33, 9, 5, 5, 6, 9, 6, 3, 66, 7, 9, 10, 4, 13
            ],
            "target_runs": [
                223, 241, 130, 166, 111, 167, 143, 209, 215, 183, 136, 148, 155, 159, 179
            ],
            "target_overs": [
                20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20
            ],
            "umpire1": [
                "Asad Rauf", "MR Benson", "Aleem Dar", "SJ Davis", "BF Bowden",
                "Aleem Dar", "IL Howell", "DJ Harper", "Asad Rauf", "Aleem Dar",
                "MR Benson", "BF Bowden", "Asad Rauf", "RE Koertzen", "BR Doctrove"
            ],
            "umpire2": [
                "RE Koertzen", "SL Shastri", "GA Pratapkumar", "DJ Harper", "K Hariharan",
                "RB Tiffin", "AM Saheba", "GA Pratapkumar", "MR Benson", "AM Saheba",
                "IL Howell", "AV Jayaprakash", "SL Shastri", "I Shivram", "RB Tiffin"
            ],
        }

    def data_to_reject(self):
        return {
            "city": [
                "Bangalore", "Chandigarh", "Delhi", "Mumbai", "Kolkata",
                "Jaipur", "Hyderabad", "Chennai", "Hyderabad", "Chandigarh",
                "Bangalore", "Chennai", "Mumbai", "Chandigarh", "Bangalore"
            ],
            "date": [
                "2008-04-18", "2008-04-19", "2008-04-19", "2008-04-20", "2008-04-20",
                "2008-04-21", "2008-04-22", "2008-04-23", "2008-04-24", "2008-04-25",
                "2008-04-26", "2008-04-26", "2008-04-27", "2008-04-27", "2008-04-28"
            ],
            "player_of_match": [
                "BB McCullum", "MEK Hussey", "MF Maharoof", "MV Boucher", "DJ Hussey",
                "SR Watson", "V Sehwag", "ML Hayden", "YK Pathan", "KC Sangakkara",
                "SR Watson", "JDP Oram", "AC Gilchrist", "SM Katich", "MS Dhoni"
            ],
            "venue": [
                "M Chinnaswamy Stadium", "Punjab Cricket Association Stadium, Mohali", "Feroz Shah Kotla",
                "Wankhede Stadium", "Eden Gardens", "Sawai Mansingh Stadium",
                "Rajiv Gandhi International Stadium, Uppal", "MA Chidambaram Stadium, Chepauk",
                "Rajiv Gandhi International Stadium, Uppal", "Punjab Cricket Association Stadium, Mohali",
                "M Chinnaswamy Stadium", "MA Chidambaram Stadium, Chepauk",
                "Dr DY Patil Sports Academy", "Punjab Cricket Association Stadium, Mohali",
                "M Chinnaswamy Stadium"
            ],
            "team1": [
                "Royal Challengers Bangalore", "Kings XI Punjab", "Delhi Daredevils", "Mumbai Indians",
                "Kolkata Knight Riders", "Rajasthan Royals", "Deccan Chargers", "Chennai Super Kings",
                "Deccan Chargers", "Kings XI Punjab", "Royal Challengers Bangalore",
                "Chennai Super Kings", "Mumbai Indians", "Kings XI Punjab", "Royal Challengers Bangalore"
            ],
            "team2": [
                "Kolkata Knight Riders", "Chennai Super Kings", "Rajasthan Royals", "Royal Challengers Bangalore",
                "Deccan Chargers", "Kings XI Punjab", "Delhi Daredevils", "Mumbai Indians",
                "Rajasthan Royals", "Mumbai Indians", "Rajasthan Royals", "Kolkata Knight Riders",
                "Deccan Chargers", "Delhi Daredevils", "Chennai Super Kings"
            ],
            "toss_winner": [
                "Royal Challengers Bangalore", "Chennai Super Kings", "Rajasthan Royals", "Mumbai Indians",
                "Deccan Chargers", "Kings XI Punjab", "Deccan Chargers", "Mumbai Indians",
                "Rajasthan Royals", "Mumbai Indians", "Rajasthan Royals", "Kolkata Knight Riders",
                "Mumbai Indians", "Kings XI Punjab", "Chennai Super Kings"
            ],
            "toss_decision": [
                "field", "bat", "bat", "bat", "bat", "bat", "bat", "field", "field", "field",
                "field", "bat", "field", "bat", "bat"
            ],
            "winner": [
                "Kolkata Knight Riders", "Chennai Super Kings", "Delhi Daredevils", "Mumbai Indians",
                "Kolkata Knight Riders", "Kings XI Punjab", "Delhi Daredevils", "Chennai Super Kings",
                "Rajasthan Royals", "Kings XI Punjab", "Rajasthan Royals", "Kolkata Knight Riders",
                "Deccan Chargers", "Delhi Daredevils", "Chennai Super Kings"
            ],
            "result": [
                "runs", "runs", "runs", "wickets", "wickets",
                "wickets", "wickets", "runs", "wickets", "runs",
                "wickets", "wickets", "wickets", "wickets", "runs"
            ],
            "result_margin": [
                140, 33, 9, 5, 5, 6, 9, 6, 3, 66, 7, 9, 10, 4, 13
            ],
            "target_runs": [
                223, 241, 130, 166, 111, 167, 143, 209, 215, 183, 136, 148, 155, 159, 179
            ],
            "target_overs": [
                20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20
            ],
            "umpire1": [
                "Asad Rauf", "MR Benson", "Aleem Dar", "SJ Davis", "BF Bowden",
                "Aleem Dar", "IL Howell", "DJ Harper", "Asad Rauf", "Aleem Dar",
                "MR Benson", "BF Bowden", "Asad Rauf", "RE Koertzen", "BR Doctrove"
            ],
            "umpire2": [
                "RE Koertzen", "SL Shastri", "GA Pratapkumar", "DJ Harper", "K Hariharan",
                "RB Tiffin", "AM Saheba", "GA Pratapkumar", "MR Benson", "AM Saheba",
                "IL Howell", "AV Jayaprakash", "SL Shastri", "I Shivram", "RB Tiffin"
            ],
        }