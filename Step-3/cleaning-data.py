import pandas as pd

matches = pd.read_csv("../matches.csv")
# Create working copies of the original datasets
matches_clean = matches.copy()

# Check for duplicate records
print("Duplicate matches:", matches_clean.duplicated().sum())
# Remove duplicate records, if present
matches_clean = matches_clean.drop_duplicates()

# Remove matches whose winner is unknown
# Only necessary for analyses requiring a known winner
matches_classification = matches_clean.dropna(subset=["winner"])

print("Matches after removing unknown winners:", matches_classification.shape)


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
