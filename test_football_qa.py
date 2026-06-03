"""
Test the upgraded ScoutAI football QA pipeline.

Default mode does not call Gemini, so it is safe for quota:
    python test_football_qa.py

To include Gemini natural-language generation:
    python test_football_qa.py --gemini
"""

import argparse
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from agent import MODEL_CHAIN
from ds_engine import build_national_strength, load_engine
from football_qa import FootballQAPipeline


QUESTIONS = [
    "מי שחקן ההתקפה הכי טוב בבאיירן מינכן?",
    "מי השחקן הכי טוב בצ'לסי?",
    "מי יכול להיות מחליף טוב עבור אנזו פרננדס?",
    "אני מחפש שחקן הגנתי אבל גם יצירתי לצ'לסי",
    "מי השחקן הכי דומה לאנזו פרננדס?",
    "מי החלוצים הכי טובים בפריימרליג?",
    "מי מלך השערים בבונדסליגה?",
    "מה תהיה התוצאה בין ברזיל לצרפת?",
    "Who is the best player in Chelsea?",
    "Who can replace Enzo Fernandez?",
    "I want a player who can be defensive but also creative, for a possession-based team, and the team is Chelsea.",
    "No, I meant a player for Chelsea who can replace Enzo Fernandez.",
    "Who is the best attacking player in La Liga?",
    "מי שחקן ההתקפה הטוב ביותר בליגה הספרדית?",
    "עזוב, מי שחקן ההתקפה הטוב ביותר בליגה הספרדית?",
    "Find a similar player to Cole Palmer.",
    "Who are the best strikers in the Premier League?",
    "Predict Brazil vs France.",
    "Compare Arsenal and Liverpool.",
    "Who is the best player in France national team?",
]


def build_llms():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return []
    return [
        ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.3,
            max_retries=0,
        )
        for model in MODEL_CHAIN
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini", action="store_true", help="Call Gemini for final answer generation.")
    args = parser.parse_args()

    load_dotenv()
    engine = load_engine()
    national_strength = build_national_strength(engine.df)
    schedule_path = "data/fwc26_match_schedule_agent.csv"
    schedule = pd.read_csv(schedule_path) if os.path.exists(schedule_path) else pd.DataFrame()
    pipeline = FootballQAPipeline(engine, national_strength, schedule, build_llms() if args.gemini else [])

    for i, question in enumerate(QUESTIONS, 1):
        result = pipeline.answerFootballQuestion(question, use_gemini=args.gemini)
        print("\n" + "=" * 90)
        print(f"TEST {i}: {question}")
        print("-" * 90)
        print(f"detected intent: {result.intent.intent}")
        print(f"matched template: {result.intent.matched_template}")
        print(
            "similarity: "
            f"TF-IDF={result.intent.tfidf_score:.3f}, "
            f"Jaccard={result.intent.jaccard_score:.3f}, "
            f"combined={result.intent.combined_score:.3f}"
        )
        print("matched entities:")
        if result.entities:
            for entity in result.entities:
                print(f"  - {entity.entity_type}: {entity.name} (score={entity.score:.3f})")
        else:
            print("  - none")
        print(f"clustering used: {result.clustering_used}")
        print(f"retrieved data count: {result.retrieved_count}")
        print(f"selected data source: {result.debug.get('selected_data_source')}")
        print(f"selected source file: {result.debug.get('selected_source_file')}")
        print(f"selected sheet: {result.debug.get('selected_sheet')}")
        print(f"context reset: {result.debug.get('context_reset')}")
        print(f"active filter applied: {result.debug.get('active_player_filter_applied')}")
        print(f"player profiles usage: {result.debug.get('player_profiles_usage')}")
        print(f"gemini used: {result.debug.get('gemini_used')}")
        print("top candidates:")
        if result.top_candidates:
            for candidate in result.top_candidates[:8]:
                print(f"  - {candidate}")
        else:
            print("  - none")
        print(f"methods: {', '.join(result.methods)}")
        if result.prompt:
            print("\nprompt preview:")
            print(result.prompt[:1200])
        print("\nfinal answer:")
        print(result.answer[:2500])


if __name__ == "__main__":
    main()
