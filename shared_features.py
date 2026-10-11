"""Leakage-safe pre-match feature engineering for IPL winner classification.

Run from any working directory: data paths are resolved relative to this file.
Every feature for a match is computed before any result from that calendar date
is added to the histories. Ball-by-ball statistics are only added to history
after their match has been played.
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MATCHES_PATH = ROOT / "matches.csv"
DELIVERIES_PATH = ROOT / "deliveries.csv"

TEAM_MAPPING = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

MATCH_REQUIRED = [
    "id", "date", "team1", "team2", "toss_winner", "toss_decision", "winner"
]

FEATURE_COLUMNS = [
    "team1_won_toss",
    "team1_bats_first",
    "prior_win_rate_diff",
    "recent_5_win_rate_diff",
    "recent_10_win_rate_diff",
    "head_to_head_win_rate_diff",
    "venue_win_rate_diff",
    "avg_runs_scored_diff",
    "avg_runs_conceded_diff",
    "avg_wickets_taken_diff",
    "avg_wickets_lost_diff",
]


def _normalise_teams(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for col in columns:
        if col in frame.columns:
            frame[col] = frame[col].replace(TEAM_MAPPING)
    return frame


def _build_delivery_summaries(deliveries_path: Path) -> dict:
    """Return {match_id: {team_name: performance dict}} from ball-by-ball data."""
    if not deliveries_path.exists():
        raise FileNotFoundError(
            f"Could not find {deliveries_path.name} in project root. "
            "Place deliveries.csv beside matches.csv."
        )

    deliveries = pd.read_csv(deliveries_path)
    needed = {
        "match_id", "batting_team", "bowling_team", "total_runs",
        "is_wicket", "extras_type"
    }
    missing = needed.difference(deliveries.columns)
    if missing:
        raise ValueError(f"deliveries.csv is missing required columns: {sorted(missing)}")

    deliveries = _normalise_teams(deliveries, ["batting_team", "bowling_team"])
    deliveries["total_runs"] = pd.to_numeric(deliveries["total_runs"], errors="coerce").fillna(0)
    deliveries["is_wicket"] = pd.to_numeric(deliveries["is_wicket"], errors="coerce").fillna(0)
    extras = deliveries["extras_type"].fillna("").astype(str).str.lower()
    deliveries["legal_ball"] = ~extras.isin(["wide", "wides", "noballs", "no ball", "noballs"])

    bat = deliveries.groupby(["match_id", "batting_team"], dropna=False).agg(
        runs_scored=("total_runs", "sum"),
        wickets_lost=("is_wicket", "sum"),
        balls_faced=("legal_ball", "sum"),
    ).reset_index().rename(columns={"batting_team": "team"})

    bowl = deliveries.groupby(["match_id", "bowling_team"], dropna=False).agg(
        runs_conceded=("total_runs", "sum"),
        wickets_taken=("is_wicket", "sum"),
    ).reset_index().rename(columns={"bowling_team": "team"})

    stats = bat.merge(bowl, on=["match_id", "team"], how="outer")
    for col in ["runs_scored", "wickets_lost", "balls_faced", "runs_conceded", "wickets_taken"]:
        stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0)

    summaries = {}
    for match_id, rows in stats.groupby("match_id", sort=False):
        team_stats = {}
        for _, row in rows.iterrows():
            team_stats[str(row["team"])] = {
                "runs_scored": float(row["runs_scored"]),
                "wickets_lost": float(row["wickets_lost"]),
                "runs_conceded": float(row["runs_conceded"]),
                "wickets_taken": float(row["wickets_taken"]),
            }
        summaries[match_id] = team_stats
    return summaries


def build_feature_dataset(
    matches_path: Path | str = MATCHES_PATH,
    deliveries_path: Path | str = DELIVERIES_PATH,
) -> pd.DataFrame:
    """Build match rows with pre-match features only, sorted chronologically."""
    matches_path = Path(matches_path)
    deliveries_path = Path(deliveries_path)
    matches = pd.read_csv(matches_path)
    missing = set(MATCH_REQUIRED).difference(matches.columns)
    if missing:
        raise ValueError(f"matches.csv is missing required columns: {sorted(missing)}")

    matches = _normalise_teams(matches, ["team1", "team2", "toss_winner", "winner"])
    matches = matches.drop_duplicates().copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="coerce")
    matches = matches.dropna(subset=["date", "team1", "team2", "winner", "toss_winner", "toss_decision"])
    matches = matches[
        matches["winner"].isin(pd.concat([matches["team1"], matches["team2"]]).unique())
    ].copy()
    matches["toss_decision"] = matches["toss_decision"].astype(str).str.strip().str.lower()
    matches = matches[matches["toss_decision"].isin(["bat", "field"])].copy()
    matches["venue"] = matches["venue"].fillna("Unknown venue").astype(str) if "venue" in matches else "Unknown venue"
    matches = matches.sort_values(["date", "id"], kind="stable").reset_index(drop=True)
    matches["team1_won"] = (matches["winner"] == matches["team1"]).astype(int)

    delivery_stats = _build_delivery_summaries(deliveries_path)

    # The list histories contain only completed matches strictly before the current date.
    results = defaultdict(list)
    venue_results = defaultdict(list)
    pair_results = defaultdict(list)  # canonical sorted team pair; 1 means pair[0] won
    performance = defaultdict(list)
    prepared = []

    for date, same_day in matches.groupby("date", sort=True):
        for _, match in same_day.iterrows():
            t1, t2 = str(match["team1"]), str(match["team2"])
            venue = str(match["venue"])
            h1, h2 = results[t1], results[t2]
            recent5_1, recent5_2 = h1[-5:], h2[-5:]
            recent10_1, recent10_2 = h1[-10:], h2[-10:]
            pair = tuple(sorted((t1, t2)))
            pair_history = pair_results[pair]
            if pair_history:
                team1_h2h_rate = (
                    sum(pair_history) / len(pair_history)
                    if t1 == pair[0]
                    else 1 - sum(pair_history) / len(pair_history)
                )
            else:
                team1_h2h_rate = 0.5

            venue1, venue2 = venue_results[(t1, venue)], venue_results[(t2, venue)]
            perf1, perf2 = performance[t1], performance[t2]

            def mean_field(history, field):
                values = [item[field] for item in history if field in item]
                return float(np.mean(values)) if values else 0.0

            toss_winner = str(match["toss_winner"])
            t1_won_toss = int(toss_winner == t1)
            t1_bats_first = int((toss_winner == t1) == (match["toss_decision"] == "bat"))

            prepared.append({
                "id": match["id"],
                "date": date,
                "team1": t1,
                "team2": t2,
                "venue": venue,
                "team1_won_toss": t1_won_toss,
                "team1_bats_first": t1_bats_first,
                "prior_win_rate_diff": (np.mean(h1) if h1 else 0.5) - (np.mean(h2) if h2 else 0.5),
                "recent_5_win_rate_diff": (np.mean(recent5_1) if recent5_1 else 0.5) - (np.mean(recent5_2) if recent5_2 else 0.5),
                "recent_10_win_rate_diff": (np.mean(recent10_1) if recent10_1 else 0.5) - (np.mean(recent10_2) if recent10_2 else 0.5),
                "head_to_head_win_rate_diff": team1_h2h_rate - 0.5,
                "venue_win_rate_diff": (np.mean(venue1) if venue1 else 0.5) - (np.mean(venue2) if venue2 else 0.5),
                # Positive differences always indicate an advantage for Team 1.
                "avg_runs_scored_diff": (mean_field(perf1, "runs_scored") - mean_field(perf2, "runs_scored")) / 100.0,
                "avg_runs_conceded_diff": (mean_field(perf2, "runs_conceded") - mean_field(perf1, "runs_conceded")) / 100.0,
                "avg_wickets_taken_diff": (mean_field(perf1, "wickets_taken") - mean_field(perf2, "wickets_taken")) / 10.0,
                "avg_wickets_lost_diff": (mean_field(perf2, "wickets_lost") - mean_field(perf1, "wickets_lost")) / 10.0,
                "team1_won": int(match["team1_won"]),
            })

        # Update all histories only after features for every match on this date are built.
        for _, match in same_day.iterrows():
            t1, t2 = str(match["team1"]), str(match["team2"])
            winner = str(match["winner"])
            t1_win = int(winner == t1)
            t2_win = int(winner == t2)
            venue = str(match["venue"])
            results[t1].append(t1_win)
            results[t2].append(t2_win)
            venue_results[(t1, venue)].append(t1_win)
            venue_results[(t2, venue)].append(t2_win)
            pair = tuple(sorted((t1, t2)))
            pair_results[pair].append(t1_win if t1 == pair[0] else t2_win)

            stats_for_match = delivery_stats.get(match["id"], {})
            for team in (t1, t2):
                if team in stats_for_match:
                    performance[team].append(stats_for_match[team])

    data = pd.DataFrame(prepared).sort_values(["date", "id"], kind="stable").reset_index(drop=True)
    if data.empty:
        raise ValueError("No usable matches remain after cleaning.")
    return data
