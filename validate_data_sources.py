"""
Validate football data sources and write:
- data/data_source_map.json
- data/player_profiles.csv
- data/data_conflicts_report.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import data_manager as dm


OUT = Path("data/data_conflicts_report.csv")


def _collect_player_sources() -> pd.DataFrame:
    frames = []

    players = dm.load_main_players()
    frames.append(pd.DataFrame({
        "player_id": players["player_id"],
        "player_name": players["name"],
        "file_name": "data/players.csv",
        "club": players["current_club_name"],
        "league": players["current_club_domestic_competition_id"],
        "season_or_date": players.get("last_season"),
        "source_priority": 1,
    }))

    if dm.PLAYER_PROFILES_FILE.exists():
        profiles = dm.load_player_profiles(regenerate_if_missing=False)
        frames.append(pd.DataFrame({
            "player_id": profiles["player_id"],
            "player_name": profiles["player_name"],
            "file_name": "data/player_profiles.csv",
            "club": profiles["club"],
            "league": profiles["league"],
            "season_or_date": profiles.get("generated_at"),
            "source_priority": 2,
        }))

    if dm.PLAYERS_CLEAN_FILE.exists():
        clean = pd.read_csv(dm.PLAYERS_CLEAN_FILE, low_memory=False)
        frames.append(pd.DataFrame({
            "player_id": clean["player_id"],
            "player_name": clean["player_name"],
            "file_name": "data/players_clean.csv",
            "club": clean["club"],
            "league": clean["league"],
            "season_or_date": "generated",
            "source_priority": 3,
        }))

    valuations = pd.read_csv(dm.DATA_DIR / "player_valuations.csv", low_memory=False)
    if not valuations.empty:
        latest = valuations.sort_values("date").groupby("player_id").tail(1)
        names = players[["player_id", "name"]]
        latest = latest.merge(names, on="player_id", how="left")
        frames.append(pd.DataFrame({
            "player_id": latest["player_id"],
            "player_name": latest["name"],
            "file_name": "data/player_valuations.csv",
            "club": latest["current_club_name"],
            "league": latest["player_club_domestic_competition_id"],
            "season_or_date": latest["date"],
            "source_priority": 4,
        }))

    return pd.concat(frames, ignore_index=True)


def build_conflict_report() -> pd.DataFrame:
    src = _collect_player_sources()
    rows = []
    for player_id, grp in src.dropna(subset=["player_id"]).groupby("player_id"):
        current = grp.sort_values("source_priority").iloc[0]
        current_club = dm.normalize_text(current["club"])
        current_league = dm.normalize_text(current["league"])
        differing = []
        for _, row in grp.iterrows():
            issue = []
            row_club = dm.normalize_text(row["club"])
            row_league = dm.normalize_text(row["league"])
            if row_club and current_club and row_club != current_club:
                issue.append("club differs from priority source")
            if row_league and current_league and row_league != current_league:
                issue.append("league differs from priority source")
            if issue:
                differing.append((row, "; ".join(issue)))
        if not differing:
            continue
        rows.append({
            "player_id": player_id,
            "player_name": current["player_name"],
            "file_name": current["file_name"],
            "club": current["club"],
            "league": current["league"],
            "season_or_date": current["season_or_date"],
            "suspected_issue": "priority source selected",
            "recommended_source_to_trust": current["file_name"],
            "recommended_club": current["club"],
            "recommended_league": current["league"],
            "source_role": "trusted_reference",
        })
        for row, issue in differing:
            rows.append({
                "player_id": player_id,
                "player_name": current["player_name"],
                "file_name": row["file_name"],
                "club": row["club"],
                "league": row["league"],
                "season_or_date": row["season_or_date"],
                "suspected_issue": issue,
                "recommended_source_to_trust": current["file_name"],
                "recommended_club": current["club"],
                "recommended_league": current["league"],
                "source_role": "conflicting_source",
            })
    return pd.DataFrame(rows)


def main():
    dm.write_source_map()
    profiles = dm.regenerate_player_profiles()
    report = build_conflict_report()
    report.to_csv(OUT, index=False)
    print(f"Regenerated {dm.PLAYER_PROFILES_FILE} ({len(profiles)} rows)")
    print(f"Wrote {dm.SOURCE_MAP_FILE}")
    print(f"Wrote {OUT} ({len(report)} rows)")
    if not report.empty:
        print(report.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
