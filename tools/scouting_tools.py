"""
Scouting tools exposed to the LLM. The model extracts intent + entities and calls the
right tool with structured arguments; the ScoutingEngine does all the ranking; the model
then narrates the returned candidate cards.
"""

from langchain.tools import tool

from scouting import (
    ScoutingEngine, SCOUT_FEATURES, parse_scouting_query, generate_scouting_response,
)


def _features(csv: str) -> list[str]:
    return [f.strip().lower() for f in (csv or "").split(",") if f.strip().lower() in SCOUT_FEATURES]


def _positions(csv: str) -> list[str]:
    return [p.strip() for p in (csv or "").split(",") if p.strip()]


def make_scouting_tools(scout: ScoutingEngine) -> list:
    """Build the four scouting tools bound to a ScoutingEngine instance."""

    @tool
    def find_similar_player(player_name: str) -> str:
        """
        Find players most SIMILAR to a reference player, using role-aware weighted
        similarity on FC26 attributes + per-90 output (not text matching). Use for
        "who is similar to X / plays like X". Pass the corrected full player name.
        """
        result, err = scout.find_similar_player(player_name)
        return err or generate_scouting_response(result)

    @tool
    def find_replacement(player_name: str, club: str = "", max_age: int = 0) -> str:
        """
        Find REPLACEMENT options for a reference player (similarity + potential + current
        ability + age fit + data reliability, same role). Use for "who can replace X".
        player_name: the player to replace (corrected full name).
        club: optional club to exclude (prefer external replacements), e.g. "Chelsea".
        max_age: optional age ceiling for the replacement (0 = no limit).
        """
        result, err = scout.find_replacement(player_name, club=club, max_age=max_age or 0)
        return err or generate_scouting_response(result)

    @tool
    def search_by_profile(role: str = "", positions: str = "", max_age: int = 0,
                          min_potential: int = 0, important_features: str = "",
                          description: str = "") -> str:
        """
        Find players matching a described PROFILE (no reference player). Use for free-text
        scouting like "a creative attacker with goal-scoring ability and high potential".
        role: one of striker, creative_attacker, winger, playmaker, box_to_box,
              defensive_midfielder, centre_back, fullback, goalkeeper (best guess from query).
        positions: optional comma list of FIFA codes (e.g. "cam,rw,st").
        max_age, min_potential: optional numeric filters.
        important_features: comma list from pace,shooting,passing,dribbling,defending,
              physic,potential,goals_per90,assists_per90.
        description: the raw user description (used to auto-fill anything left blank).
        """
        feats = _features(important_features)
        if description and not (role or feats):
            ctx = parse_scouting_query(description)
            role = role or ctx.get("role", "")
            feats = feats or ctx.get("important_features", [])
            max_age = max_age or ctx.get("age_max", 0)
            min_potential = min_potential or ctx.get("potential_min", 0)
        result, err = scout.search_by_profile(
            role=role, positions=_positions(positions), age_max=max_age or 0,
            potential_min=min_potential or 0, important_features=feats,
        )
        return err or generate_scouting_response(result)

    @tool
    def find_wonderkids(role: str = "", positions: str = "", max_age: int = 21,
                        min_potential: int = 80, important_features: str = "") -> str:
        """
        Find young high-POTENTIAL prospects (wonderkids). Use for "best young prospects /
        wonderkids", optionally by role/position. Ranking is potential-led.
        role: optional role (e.g. centre_back, striker, winger).
        positions: optional comma list of FIFA codes.
        max_age: age ceiling (default 21). min_potential: potential floor (default 80).
        important_features: optional comma list of features to emphasise.
        """
        result, err = scout.find_wonderkids(
            role=role, positions=_positions(positions), age_max=max_age or 21,
            potential_min=min_potential or 80, important_features=_features(important_features),
        )
        return err or generate_scouting_response(result)

    return [find_similar_player, find_replacement, search_by_profile, find_wonderkids]
