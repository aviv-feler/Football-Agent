"""
Verification for the scouting country/nationality/league filter fix.
Runs the real pipeline: parse_scouting_query -> ScoutingEngine.search_by_profile,
exactly as the deterministic router / tool layer calls it, and checks that every
returned player matches the requested country (nationality) OR league.
"""
import sys
from ds_engine import load_engine
from scouting import (ScoutingEngine, parse_scouting_query, COUNTRY_LEAGUE,
                      LEAGUE_NAME_TO_CODE, LEAGUE_CODE_COUNTRY)

engine = load_engine()
scout = ScoutingEngine(engine)


def expected_codes_and_nats(country: str):
    """Given the parsed `country` value, return (allowed_league_codes, allowed_nationalities)."""
    low = country.lower()
    if low in LEAGUE_NAME_TO_CODE:          # explicit league -> league only
        return {LEAGUE_NAME_TO_CODE[low]}, set()
    if country.upper() in LEAGUE_CODE_COUNTRY:
        return {country.upper()}, set()
    # country/demonym -> nationality OR domestic league
    from scouting import normalize_country
    nat = normalize_country(country)
    codes = {COUNTRY_LEAGUE[nat]} if nat in COUNTRY_LEAGUE else set()
    return codes, {nat}


TESTS = [
    ("attacking player from Italy", {}),
    ("fast winger from Brazil", {}),
    ("defender from the Premier League", {}),
    ("young striker from Spain under 23", {}),
    ("midfielder from Germany", {}),
]

rows = []
all_ok = True
for query, _ in TESTS:
    ctx = parse_scouting_query(query)
    result, err = scout.search_by_profile(
        role=ctx["role"], age_max=ctx["age_max"], potential_min=ctx["potential_min"],
        important_features=ctx["important_features"], country=ctx["country"],
        position_group=ctx["position_group"], limit=5,
    )
    print("=" * 90)
    print(f"QUERY: {query!r}")
    print(f"  parsed -> role={ctx['role']!r} group={ctx['position_group']!r} "
          f"country={ctx['country']!r} age_max={ctx['age_max']} pot_min={ctx['potential_min']} "
          f"feats={ctx['important_features']}")
    if err:
        print("  ERROR:", err)
        all_ok = False
        rows.append((query, "—", "ERROR", "❌"))
        continue

    codes, nats = expected_codes_and_nats(ctx["country"])
    age_cap = ctx["age_max"] or None
    want_attack = ctx["position_group"] == "Attack" or ctx["role"] in ("striker", "winger", "creative_attacker")

    for c in result["candidates"]:
        league = c["league"]   # exact league of THIS candidate (no name-collision lookup)
        nat = c["nationality"]
        pos = c["position"]
        age = c["age"]
        region_ok = (nat in nats) or (league in codes) if (nats or codes) else True
        age_ok = (age_cap is None) or (age < age_cap)
        ok = region_ok and age_ok
        all_ok = all_ok and ok
        mark = "✅" if ok else "❌"
        print(f"   {mark} {c['player_name']:<26} | nat={nat:<16} | league={league:<5} | {pos:<16} | age {age}")
        rows.append((query, c["player_name"], f"{nat}/{league}", mark))

print("=" * 90)
print("\nSUMMARY TABLE: Query | Player | Nationality/League | Match?")
print("-" * 90)
last_q = None
for q, player, natleague, mark in rows:
    qcol = q if q != last_q else ""
    print(f"{qcol:<34} | {player:<26} | {natleague:<22} | {mark}")
    last_q = q

print("\nALL TESTS PASSED ✅" if all_ok else "\nSOME TESTS FAILED ❌")
sys.exit(0 if all_ok else 1)
