import pandas as pd

matches = pd.read_csv("../matches.csv")
deliveries = pd.read_csv("../deliveries.csv")

"""
<name>.shape 
returns the dimension of your dataframe ( no. of rows, no. of columns )
dataframe being akin to a table present in excel or google sheets
"""

print("Matches shape:", matches.shape)
print("Deliveries shape:", deliveries.shape)

print("\nMatch columns:")
print(matches.columns.tolist())

print("\nDelivery columns:")
print(deliveries.columns.tolist())


"""
.isnull(): Checks every cell in the matches DataFrame and returns a new table of the same size with True where data is missing and False where data is present.

.sum(): Adds up those True and False values down each column (treating True as 1 and False as 0).
"""
print("\nMissing values in matches:")
print(matches.isnull().sum())

print("\nMissing values in deliveries:")
print(deliveries.isnull().sum())


# Inspect column data types
print("Match data types:")
print(matches.dtypes)

print("\nDelivery data types:")
print(deliveries.dtypes)


# Inspect team names across both datasets
match_teams = set(matches["team1"]) | set(matches["team2"])
delivery_teams = set(deliveries["batting_team"]) | set(deliveries["bowling_team"])

print("\nTeams in matches.csv:")
print(sorted(match_teams))

print("\nTeams in deliveries.csv:")
print(sorted(delivery_teams))

print("\nTeams only in matches.csv:")
print(sorted(match_teams - delivery_teams))

print("\nTeams only in deliveries.csv:")
print(sorted(delivery_teams - match_teams))


# Verify match ID relationships
match_ids = set(matches["id"])
delivery_match_ids = set(deliveries["match_id"])

print(
    "\nDelivery match IDs absent from matches.csv:", len(delivery_match_ids - match_ids)
)

print("Matches without deliveries:", len(match_ids - delivery_match_ids))
