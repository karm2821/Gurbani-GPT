#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_test_comprehensive.py — Comprehensive Gurbani GPT Test Suite
Tests: retrieval quality, concept expansion, confidence detection,
       real-life situations, Roman Punjabi, hallucination detection.
Run: python scripts/05_test_comprehensive.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rag_engine import GurbaniRAG, CONCEPT_MAP, TIER1_THRESHOLD, TIER2_THRESHOLD

CHROMA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')

# ─────────────────────────────────────────────────────────────────────────────
# TEST CASES
# ─────────────────────────────────────────────────────────────────────────────

DIRECT_TOPIC_TESTS = [
    # (test_name, query, expected_concept_keywords)
    ("Anger — direct",        "How can I overcome anger according to Gurbani?",        ["anger", "gussa", "krodh"]),
    ("Ego — direct",          "What does Gurbani say about ego?",                      ["ego", "haumai"]),
    ("Greed — direct",        "Gurbani on greed and materialism",                      ["greed", "lobh"]),
    ("Attachment — direct",   "What does Gurbani teach about attachment?",             ["attachment", "moh"]),
    ("Fear — direct",         "How to overcome fear according to Gurbani?",            ["fear"]),
    ("Sadness — direct",      "Gurbani on sadness and sorrow",                         ["sadness", "dukh"]),
    ("Jealousy — direct",     "What does Gurbani say about jealousy?",                 ["jealousy"]),
    ("Forgiveness — direct",  "How to forgive someone according to Gurbani?",          ["forgiveness", "forgive"]),
    ("Humility — direct",     "What does Gurbani teach about humility?",               ["humility"]),
    ("Contentment — direct",  "Gurbani on contentment and satisfaction",               ["contentment", "santokh"]),
    ("Naam — direct",         "Importance of Naam according to Gurbani",               ["naam"]),
    ("Hukam — direct",        "What is Hukam in Gurbani?",                             ["hukam"]),
    ("Death — direct",        "What does Gurbani say about death?",                    ["death"]),
    ("Peace — direct",        "How to find inner peace through Gurbani?",              ["peace"]),
]

REAL_LIFE_TESTS = [
    ("Betrayal",          "My close friend betrayed me. What should I do?"),
    ("Family conflict",   "I always fight with my parents. How can Gurbani help?"),
    ("Insult",            "Someone insulted me in front of everyone. I feel humiliated."),
    ("Exam failure",      "I failed my exam and feel completely useless and worthless."),
    ("Jealous of friend", "I am jealous of my friend who is more successful than me."),
    ("Worry about future","I can't stop worrying about my future and career."),
    ("Attachment",        "I am too attached to someone and it is causing me pain."),
    ("Life purpose",      "I don't know what to do with my life. What is my purpose?"),
    ("Financial stress",  "I am under a lot of financial stress and anxiety about money."),
    ("Loneliness",        "I feel very lonely even when surrounded by people."),
    ("Anger at spouse",   "I get very angry at my husband/wife. How can Gurbani guide me?"),
    ("Grief",             "I lost someone very close to me. How do I deal with grief?"),
]

LANGUAGE_TESTS = [
    # (test_name, query)
    ("Roman Punjabi — gussa",         "mere ko bahut gussa aunda hai"),
    ("Roman Punjabi — haumai",        "haumai kaise khatam karen"),
    ("Mixed Punjabi-English — anger", "main bahut angry haan, koi Gurbani batao"),
    ("Gurmukhi query",                "ਮੇਰਾ ਮਨ ਬਹੁਤ ਦੁਖੀ ਹੈ"),
    ("Hindi transliteration",         "lobh aur maya se kaise mukt hoon"),
    ("Misspelled concept",            "how to reduce kroddh and haumaai"),
    ("English casual",                "gurbani says what about ego and pride"),
]

HALLUCINATION_TESTS = [
    # Questions for which the DB likely has no strong direct reference
    # Verify model says 'supporting/general' rather than fabricating
    ("Very specific — no DB match 1", "What does Gurbani say about using social media?"),
    ("Very specific — no DB match 2", "What does Gurbani say about cryptocurrency and investing?"),
    ("Unusual topic",                 "Does Gurbani mention anything about sleep deprivation?"),
    ("Very niche",                    "What is the Gurbani teaching specifically about jealousy of a sibling?"),
]

FOLLOWUP_TESTS = [
    # Simulate a follow-up conversation
    {
        "name": "Follow-up: anger → other person wrong",
        "history": [
            {"role": "user",      "content": "How can I control my anger according to Gurbani?"},
            {"role": "assistant", "content": "[Previous answer about ਕ੍ਰੋਧ and ਸਹਜ]"},
        ],
        "followup": "But what if the other person is actually wrong and deserves my anger?",
    },
    {
        "name": "Follow-up: grief → how long",
        "history": [
            {"role": "user",      "content": "I lost my father recently. How does Gurbani comfort us?"},
            {"role": "assistant", "content": "[Previous answer about Hukam and acceptance]"},
        ],
        "followup": "How long does this pain last? Will it ever go away?",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"
INFO = "ℹ️  INFO"

def sep(char="─", n=70):
    print(char * n)

def run_retrieval_test(rag, name, query, expected_concepts=None):
    """Run a retrieval test and print detailed results."""
    print(f"\n  📋 {name}")
    print(f"  Query: {query}")

    expansion = rag.expand_query(query)
    passages  = rag.retrieve(query, n=5)
    confidence = rag.detect_confidence(passages)

    matched = expansion["matched_concepts"]
    terms   = expansion["gurbani_terms"]

    # Check concept matching
    if expected_concepts:
        found = any(c in matched for c in expected_concepts)
        concept_status = PASS if found else FAIL
        print(f"  Concept match: {concept_status} — matched: {matched} (expected any of: {expected_concepts})")
    else:
        print(f"  Concepts matched: {matched or '(none)'}")

    print(f"  Gurbani terms: {terms[:6]}")
    print(f"  Confidence: {confidence}")
    print(f"  Search queries used: {expansion['search_queries']}")

    if passages:
        print(f"  Top passages:")
        for i, p in enumerate(passages[:3], 1):
            tier = rag._tier_label(p['relevance'])
            tier_icon = {"DIRECT": "🟢", "SUPPORTING": "🟡", "GENERAL": "🔴"}.get(tier, "⚪")
            print(f"    {i}. {tier_icon} [{tier}] Ang {p['ang']} | {p['raag']} | {p['author']}")
            print(f"       Relevance: {p['relevance']}")
            print(f"       Preview: {p['text'][:100].replace(chr(10), ' ')}...")
    else:
        print(f"  {WARN} No passages retrieved!")

    return passages, confidence, expansion


def run_answer_test(rag, name, query, history=None, full_answer=False):
    """Run a full answer test (non-streaming)."""
    print(f"\n  💬 {name}")
    print(f"  Query: {query}")

    passages   = rag.retrieve(query, n=5)
    confidence = rag.detect_confidence(passages)
    print(f"  Confidence: {confidence}")

    if not passages:
        print(f"  {WARN} No passages retrieved — answer may be poor quality")
        return

    answer = rag.answer(query, passages, history=history)

    # Check for refusal pattern (should not happen for LOW confidence either)
    refused = "don't directly address" in answer.lower() and "please search" in answer.lower()
    if refused:
        print(f"  {WARN} System gave a refusal response — check prompt")
    else:
        print(f"  {PASS} System provided an answer")

    # Check for hallucination signals (rough heuristic)
    suspicious_phrases = [
        "gurbani says that anger is caused by",
        "guru nanak said that you should",
        "gurbani explicitly states that",
    ]
    for phrase in suspicious_phrases:
        if phrase in answer.lower():
            print(f"  {WARN} Possible overstatement detected: '{phrase}'")

    # Check answer structure
    has_gurbani_ref = "📖" in answer or "Gurbani:" in answer or "Ang" in answer
    has_application = "🌱" in answer or "applying" in answer.lower() or "apply" in answer.lower()
    print(f"  Has Gurbani reference section: {'Yes' if has_gurbani_ref else 'No'}")
    print(f"  Has application section: {'Yes' if has_application else 'No'}")

    if full_answer:
        print(f"\n  ─── Answer ───")
        print(answer[:800] + ("..." if len(answer) > 800 else ""))
        print(f"  ─────────────")

    return answer


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print()
    sep("═")
    print("   🙏  Gurbani GPT — Comprehensive Test Suite")
    sep("═")
    print()

    rag = GurbaniRAG(chroma_dir=CHROMA_DIR)

    if not rag.ready():
        print("  ❌ ChromaDB is empty. Run 03_embed_store.py first.")
        return

    print(f"  ✅ Gurbani DB ready — {rag.count()} Shabads indexed")
    print()

    # ── 1. CONCEPT MAP COVERAGE CHECK ────────────────────────────────────────
    sep()
    print("  1️⃣  CONCEPT MAP COVERAGE")
    sep()
    print(f"  Total concepts in map: {len(CONCEPT_MAP)}")
    print(f"  Sample concepts: {list(CONCEPT_MAP.keys())[:10]}")
    print()

    # ── 2. DIRECT TOPIC RETRIEVAL TESTS ──────────────────────────────────────
    sep()
    print("  2️⃣  DIRECT TOPIC RETRIEVAL TESTS")
    sep()
    direct_passed = 0
    for name, query, expected in DIRECT_TOPIC_TESTS:
        passages, confidence, expansion = run_retrieval_test(rag, name, query, expected)
        matched = expansion["matched_concepts"]
        if any(c in matched for c in expected):
            direct_passed += 1
        print()

    print(f"\n  Direct topic concept match: {direct_passed}/{len(DIRECT_TOPIC_TESTS)}")
    sep()

    # ── 3. REAL LIFE SITUATION TESTS (Retrieval only) ─────────────────────────
    sep()
    print("  3️⃣  REAL-LIFE SITUATION RETRIEVAL TESTS")
    sep()
    for name, query in REAL_LIFE_TESTS:
        run_retrieval_test(rag, name, query)
        print()
    sep()

    # ── 4. LANGUAGE TESTS ────────────────────────────────────────────────────
    sep()
    print("  4️⃣  LANGUAGE / ROMAN PUNJABI TESTS")
    sep()
    for name, query in LANGUAGE_TESTS:
        run_retrieval_test(rag, name, query)
        print()
    sep()

    # ── 5. FULL ANSWER TESTS (select key ones) ───────────────────────────────
    sep()
    print("  5️⃣  FULL ANSWER QUALITY TESTS (key questions)")
    sep()
    answer_tests = [
        ("Anger — full answer",    "I am struggling with a lot of anger. According to Gurbani, how can I overcome it?"),
        ("Ego — full answer",      "How does Gurbani describe ego and how can I overcome it?"),
        ("Failure — full answer",  "I failed my exam and feel completely useless. What does Gurbani say?"),
        ("Betrayal — full answer", "My best friend betrayed my trust. How should I deal with this according to Gurbani?"),
    ]
    for name, query in answer_tests:
        run_answer_test(rag, name, query, full_answer=True)
        print()
    sep()

    # ── 6. HALLUCINATION DETECTION TESTS ─────────────────────────────────────
    sep()
    print("  6️⃣  HALLUCINATION DETECTION TESTS")
    print("  (Check that LOW confidence answers are honest, not fabricated)")
    sep()
    for name, query in HALLUCINATION_TESTS:
        passages   = rag.retrieve(query, n=5)
        confidence = rag.detect_confidence(passages)
        print(f"\n  📋 {name}")
        print(f"  Query: {query}")
        print(f"  Confidence: {confidence}")
        if confidence == "LOW":
            print(f"  {PASS} Correctly identified as LOW confidence")
        elif confidence == "MEDIUM":
            print(f"  {INFO} MEDIUM confidence — answer should note limited direct match")
        else:
            print(f"  {INFO} HIGH confidence — check if passage is genuinely relevant")
        if passages:
            p = passages[0]
            tier = rag._tier_label(p['relevance'])
            print(f"  Best match: [{tier}] Ang {p['ang']} Relevance: {p['relevance']}")
        print()
    sep()

    # ── 7. FOLLOW-UP / CONVERSATION TESTS ───────────────────────────────────
    sep()
    print("  7️⃣  FOLLOW-UP QUESTION TESTS")
    sep()
    for test in FOLLOWUP_TESTS:
        run_answer_test(rag, test["name"], test["followup"],
                        history=test["history"], full_answer=True)
        print()
    sep()

    print()
    print("  ✅ Test suite complete.")
    print()
    print("  SUMMARY OF WHAT TO VERIFY MANUALLY:")
    print("  1. Anger questions → should show ਕ੍ਰੋਧ-related passages as DIRECT")
    print("  2. Roman Punjabi (gussa) → should map to anger/ਕ੍ਰੋਧ concepts")
    print("  3. Hallucination tests → LOW confidence should be transparent in answers")
    print("  4. Follow-up tests → second answer should reference first conversation")
    print("  5. Answers should clearly separate: Gurbani text / Meaning / Application")
    print()


if __name__ == '__main__':
    main()
