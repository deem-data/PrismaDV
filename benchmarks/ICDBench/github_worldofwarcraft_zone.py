from .abstract_base_case import AbstractBaseCase

# Code and data from https://github.com/davide97l/WoW-dataset-analysis
class GithubWorldOfWarcraftZone(AbstractBaseCase):

    def sample_data(self):
        return {
            "char": [59425, 65494, 65325, 65490, 2288, 2289, 61239, 59772, 22937, 23062,
                     48432, 582, 33256, 22307, 22466, 1003, 62, 2663, 49944],
            "level": [1, 9, 14, 18, 60, 60, 68, 69, 69, 69,
                      70, 70, 70, 70, 70, 70, 70, 70, 70],
            "race": ["Orc"] * 19,
            "charclass": ["Rogue", "Hunter", "Warrior", "Hunter", "Hunter", "Hunter",
                          "Hunter", "Warrior", "Rogue", "Shaman",
                          "Warrior", "Warrior", "Warrior", "Warrior", "Warrior",
                          "Warrior", "Warrior", "Warrior", "Warrior"],
            "zone": ["Orgrimmar", "Durotar", "Ghostlands", "Ghostlands", "Hellfire Peninsula",
                     "Hellfire Peninsula", "Blade's Edge Mountains", "Shadowmoon Valley", "Warsong Gulch",
                     "Shattrath City", "Blade's Edge Mountains", "Sethekk Halls", "Orgrimmar", "Orgrimmar",
                     "Undercity", "Tirisfal Glades", "Shattrath City", "Shattrath City", "Terokkar Forest"],
            "guild": [165, -1, -1, -1, -1, -1, 243, 35, 243, 103,
                      79, 19, 53, 174, 101, 204, 5, 53, 167],
            "timestamp": ["01/01/08 00:02:04", "01/01/08 00:02:04", "01/01/08 00:02:04", "01/01/08 00:02:04",
                          "01/01/08 00:02:09", "01/01/08 00:02:09", "01/01/08 00:02:14", "01/01/08 00:02:14",
                          "01/01/08 00:02:14", "01/01/08 00:02:14", "01/01/08 00:02:19", "01/01/08 00:02:19",
                          "01/01/08 00:02:19", "01/01/08 00:02:19", "01/01/08 00:02:19", "01/01/08 00:02:19",
                          "01/01/08 00:02:19", "01/01/08 00:02:19", "01/01/08 00:02:19"]
        }

    def downstream_code(self):
        return """
            wowah['zone'].replace({'Dalaran競技場': 'Dalaran Arena'}, inplace=True)
            """

    def assumption_in_natural_language(self):
        return "The values in the zone column are one of the following: 'Blade\'s Edge Mountains', 'Durotar', 'Ghostlands', 'Hellfire Peninsula', 'Orgrimmar', 'Sethekk Halls', 'Shadowmoon Valley', 'Shattrath City', 'Terokkar Forest', 'Tirisfal Glades', 'Undercity', 'Warsong Gulch', 'Dalaran Arena', 'Dalaran競技場'."

    def target_column(self):
        return "zone"

    def ground_truth_constraint(self):
        return ".isContainedIn('zone', ['Blade\'s Edge Mountains', 'Durotar', 'Ghostlands', 'Hellfire Peninsula', 'Orgrimmar', 'Sethekk Halls', 'Shadowmoon Valley', 'Shattrath City', 'Terokkar Forest', 'Tirisfal Glades', 'Undercity', 'Warsong Gulch', 'Dalaran Arena', 'Dalaran競技場'])"

    def data_to_pass(self):
        return {
            "char": [59425, 65494, 65325, 65490, 2288, 2289, 61239, 59772, 22937, 23062,
                     48432, 582, 33256, 22307, 22466, 1003, 62, 2663, 49944],
            "level": [1, 9, 14, 18, 60, 60, 68, 69, 69, 69,
                      70, 70, 70, 70, 70, 70, 70, 70, 70],
            "race": ["Orc"] * 19,
            "charclass": ["Rogue", "Hunter", "Warrior", "Hunter", "Hunter", "Hunter",
                          "Hunter", "Warrior", "Rogue", "Shaman",
                          "Warrior", "Warrior", "Warrior", "Warrior", "Warrior",
                          "Warrior", "Warrior", "Warrior", "Warrior"],
            "zone": ["Dalaran競技場", "Dalaran Arena", "Ghostlands", "Ghostlands", "Hellfire Peninsula",
                     "Hellfire Peninsula", "Blade's Edge Mountains", "Shadowmoon Valley", "Warsong Gulch",
                     "Shattrath City", "Blade's Edge Mountains", "Sethekk Halls", "Orgrimmar", "Orgrimmar",
                     "Undercity", "Tirisfal Glades", "Shattrath City", "Shattrath City", "Terokkar Forest"],
            "guild": [165, -1, -1, -1, -1, -1, 243, 35, 243, 103,
                      79, 19, 53, 174, 101, 204, 5, 53, 167],
            "timestamp": ["01/01/08 00:02:04", "01/01/08 00:02:04", "01/01/08 00:02:04", "01/01/08 00:02:04",
                          "01/01/08 00:02:09", "01/01/08 00:02:09", "01/01/08 00:02:14", "01/01/08 00:02:14",
                          "01/01/08 00:02:14", "01/01/08 00:02:14", "01/01/08 00:02:19", "01/01/08 00:02:19",
                          "01/01/08 00:02:19", "01/01/08 00:02:19", "01/01/08 00:02:19", "01/01/08 00:02:19",
                          "01/01/08 00:02:19", "01/01/08 00:02:19", "01/01/08 00:02:19"]
        }

    def data_to_reject(self):
        return {
            "char": [59425, 65494, 65325, 65490, 2288, 2289, 61239, 59772, 22937, 23062,
                     48432, 582, 33256, 22307, 22466, 1003, 62, 2663, 49944],
            "level": [1, 9, 14, 18, 60, 60, 68, 69, 69, 69,
                      70, 70, 70, 70, 70, 70, 70, 70, 70],
            "race": ["Orc"] * 19,
            "charclass": ["Rogue", "Hunter", "Warrior", "Hunter", "Hunter", "Hunter",
                          "Hunter", "Warrior", "Rogue", "Shaman",
                          "Warrior", "Warrior", "Warrior", "Warrior", "Warrior",
                          "Warrior", "Warrior", "Warrior", "Warrior"],
            "zone": ["Dalaran競技場", "Dalaran Arena", "Non existing zone", "Ghostlands", "Hellfire Peninsula",
                     "Hellfire Peninsula", "Blade's Edge Mountains", "Shadowmoon Valley", "Warsong Gulch",
                     "Shattrath City", "Blade's Edge Mountains", "Sethekk Halls", "Orgrimmar", "Orgrimmar",
                     "Undercity", "Tirisfal Glades", "Shattrath City", "Shattrath City", "Terokkar Forest"],
            "guild": [165, -1, -1, -1, -1, -1, 243, 35, 243, 103,
                      79, 19, 53, 174, 101, 204, 5, 53, 167],
            "timestamp": ["01/01/08 00:02:04", "01/01/08 00:02:04", "01/01/08 00:02:04", "01/01/08 00:02:04",
                          "01/01/08 00:02:09", "01/01/08 00:02:09", "01/01/08 00:02:14", "01/01/08 00:02:14",
                          "01/01/08 00:02:14", "01/01/08 00:02:14", "01/01/08 00:02:19", "01/01/08 00:02:19",
                          "01/01/08 00:02:19", "01/01/08 00:02:19", "01/01/08 00:02:19", "01/01/08 00:02:19",
                          "01/01/08 00:02:19", "01/01/08 00:02:19", "01/01/08 00:02:19"]
        }