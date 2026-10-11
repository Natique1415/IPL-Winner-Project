from pathlib import Path

import pandas as pd

# Project root is the parent of Step-3.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATCHES_PATH = PROJECT_ROOT / "matches.csv"
DELIVERIES_PATH = PROJECT_ROOT / "deliveries.csv"

TEAM_MAPPING = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}


def standardize_team_names(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Standardize known historical franchise names in a working copy."""
    data = data.copy()
    for column in columns:
        if column in data.columns:
            data[column] = data[column].replace(TEAM_MAPPING)
    return data


def main() -> None:
    if not MATCHES_PATH.exists():
        raise FileNotFoundError(f"Could not find matches.csv at {MATCHES_PATH}.")
    if not DELIVERIES_PATH.exists():
        raise FileNotFoundError(f"Could not find deliveries.csv at {DELIVERIES_PATH}.")

    # Read the raw data and work on copies; do not overwrite source CSVs.
    matches_raw = pd.read_csv(MATCHES_PATH)
    deliveries_raw = pd.read_csv(DELIVERIES_PATH)

    print("Raw matches shape:", matches_raw.shape)
    print("Raw deliveries shape:", deliveries_raw.shape)

    print("\nDuplicate match rows:", matches_raw.duplicated().sum())
    print("Duplicate delivery rows:", deliveries_raw.duplicated().sum())

    matches_clean = matches_raw.drop_duplicates().copy()
    deliveries_clean = deliveries_raw.drop_duplicates().copy()

    # Apply the same franchise mapping to both datasets so team-level
    # features derived from deliveries.csv use the same names as matches.csv.
    matches_clean = standardize_team_names(
        matches_clean, ["team1", "team2", "toss_winner", "winner"]
    )
    deliveries_clean = standardize_team_names(
        deliveries_clean, ["batting_team", "bowling_team"]
    )

    # Keep all cleaned matches for general analysis, and make a separate
    # subset for classification tasks that require a known winner.
    matches_classification = matches_clean.dropna(subset=["winner"]).copy()

    print("\nMatches after duplicate removal:", matches_clean.shape)
    print("Matches with known winners for classification:", matches_classification.shape)
    print("Deliveries after duplicate removal:", deliveries_clean.shape)

    if "id" in matches_clean.columns and "match_id" in deliveries_clean.columns:
        match_ids = set(matches_clean["id"].dropna())
        delivery_match_ids = set(deliveries_clean["match_id"].dropna())
        print("\nMatch IDs without delivery records:", len(match_ids - delivery_match_ids))
        print("Delivery match IDs absent from matches.csv:", len(delivery_match_ids - match_ids))

    print(
        "\nCleaning completed in memory. Original matches.csv and deliveries.csv "
        "were not modified or overwritten."
    )
    print(
        "The classification scripts and shared feature module must apply the same "
        "cleaning rules when loading the raw files."
    )


if __name__ == "__main__":
    main()
