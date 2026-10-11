from pathlib import Path

import pandas as pd

# Project root is the parent of Step-2.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATCHES_PATH = PROJECT_ROOT / "matches.csv"
DELIVERIES_PATH = PROJECT_ROOT / "deliveries.csv"


def inspect_dataset(name: str, path: Path) -> pd.DataFrame:
    """Load a CSV and print a concise overview of its contents."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.name} at {path}. "
            "Place the dataset in the project root."
        )

    data = pd.read_csv(path)

    print(f"\n{'=' * 60}")
    print(f"{name} OVERVIEW")
    print(f"{'=' * 60}")
    print(f"Shape (rows, columns): {data.shape}")

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nMissing values by column:")
    missing = data.isna().sum()
    missing = missing[missing > 0]
    print(missing.to_string() if not missing.empty else "No missing values.")

    print("\nData types:")
    print(data.dtypes.to_string())

    print(f"\nDuplicate rows: {data.duplicated().sum()}")

    return data


def main() -> None:
    matches = inspect_dataset("MATCHES", MATCHES_PATH)
    deliveries = inspect_dataset("DELIVERIES", DELIVERIES_PATH)

    print(f"\n{'=' * 60}")
    print("CROSS-DATASET CHECKS")
    print(f"{'=' * 60}")

    if "id" in matches.columns and "match_id" in deliveries.columns:
        match_ids = set(matches["id"].dropna())
        delivery_match_ids = set(deliveries["match_id"].dropna())

        print(f"Unique match IDs in matches.csv: {len(match_ids)}")
        print(f"Unique match IDs in deliveries.csv: {len(delivery_match_ids)}")
        print(f"Match IDs without delivery records: {len(match_ids - delivery_match_ids)}")
        print(f"Delivery match IDs absent from matches.csv: {len(delivery_match_ids - match_ids)}")
    else:
        print("Could not compare match IDs: expected 'id' and 'match_id' columns.")

    if {"team1", "team2"}.issubset(matches.columns):
        team_counts = pd.concat(
            [matches["team1"], matches["team2"]],
            ignore_index=True,
        ).value_counts(dropna=False)
        print("\nTeam appearances (raw franchise names):")
        print(team_counts.sort_index().to_string())


if __name__ == "__main__":
    main()
