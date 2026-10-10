import pandas as pd

matches = pd.read_csv("../matches.csv")
deliveries = pd.read_csv("../deliveries.csv")


team_counts = pd.concat([matches["team1"], matches["team2"]]).value_counts()

print("Matches per team name:")
print(team_counts.sort_index())

teams = sorted(set(matches["team1"]) | set(matches["team2"]))

# Display seasons in which each team name appears
for team in teams:
    team_matches = matches[(matches["team1"] == team) | (matches["team2"] == team)]

    seasons = sorted(team_matches["season"].astype(str).unique())

    print(f"{team}: {', '.join(seasons)}")
