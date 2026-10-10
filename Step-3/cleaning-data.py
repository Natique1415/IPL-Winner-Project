import pandas as pd

matches = pd.read_csv("../matches.csv")
deliveries = pd.read_csv("../deliveries.csv")
# Create working copies of the original datasets
matches_clean = matches.copy()
deliveries_clean = deliveries.copy()

# Check for duplicate records
print("Duplicate matches:", matches_clean.duplicated().sum())
print("Duplicate deliveries:", deliveries_clean.duplicated().sum())

# Remove duplicate records, if present
matches_clean = matches_clean.drop_duplicates()
deliveries_clean = deliveries_clean.drop_duplicates()

# Remove matches whose winner is unknown
# Only necessary for analyses requiring a known winner
matches_classification = matches_clean.dropna(subset=["winner"])

print("Matches after removing unknown winners:", matches_classification.shape)
print("Deliveries retained:", deliveries_clean.shape)


# Standardize historical franchise names
team_mapping = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

# Apply mappings to match data
for column in ["team1", "team2", "toss_winner", "winner"]:
    matches_clean[column] = matches_clean[column].replace(team_mapping)

# Apply mappings to delivery data
for column in ["batting_team", "bowling_team"]:
    deliveries_clean[column] = deliveries_clean[column].replace(team_mapping)
