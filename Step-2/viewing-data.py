import pandas as pd

matches = pd.read_csv("../matches.csv")


"""
<name>.shape 
returns the dimension of your dataframe ( no. of rows, no. of columns )
dataframe being akin to a table present in excel or google sheets
"""

print("Matches shape:", matches.shape)

print("\nMatch columns:")
print(matches.columns.tolist())


"""
.isnull(): Checks every cell in the matches DataFrame and returns a new table of the same size with True where data is missing and False where data is present.

.sum(): Adds up those True and False values down each column (treating True as 1 and False as 0).
"""
print("\nMissing values in matches:")
print(matches.isnull().sum())


# Inspect column data types
print("Match data types:")
print(matches.dtypes)
