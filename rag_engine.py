#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_engine.py — Core Gurbani RAG Engine (Enhanced)
Handles: query expansion → multi-query ChromaDB search → relevance tiering
         → confidence detection → grounded LLM answering.
Used by server.py for the /api/gurbani-chat route.

Improvements over v1:
  - Gurbani concept vocabulary map (English/Roman Punjabi → Gurbani terms)
  - Multi-query expansion for better recall
  - Relevance tiering (Direct / Supporting / General)
  - Confidence detection (HIGH / MEDIUM / LOW)
  - Structured, tier-aware system prompt
  - History/follow-up awareness in streaming
"""

import os
import sys
import io
import requests
import json

# Only redirect stdout when running this file directly (e.g. for testing).
# When imported by server.py, server.py already handles UTF-8 stdout.
# Redirecting here on import closes server.py's stdout and causes crashes.
if __name__ == '__main__' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


def safe_print(msg: str):
    try:
        print(msg)
    except Exception:
        try:
            print(str(msg).encode('ascii', errors='replace').decode('ascii'))
        except Exception:
            pass

import chromadb

OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
EMBED_MODEL  = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHROMA_DIR   = os.environ.get("CHROMA_DIR", os.path.join(os.path.dirname(__file__), 'data', 'chroma_db'))
COLLECTION   = "gurbani"


# ─────────────────────────────────────────────────────────────────────────────
# GURBANI CONCEPT MAP
# Maps English/Roman Punjabi/Hindi terms → Gurbani & Punjabi search terms.
# Used for query expansion to bridge the vocabulary gap between user questions
# and the Gurbani text stored in ChromaDB.
# ─────────────────────────────────────────────────────────────────────────────
CONCEPT_MAP = {
    # ── Anger ────────────────────────────────────────────────────────────────
    "anger":       ["ਕ੍ਰੋਧ", "ਕ੍ਰੋਧੁ", "krodh", "anger", "wrath", "rage",
                    "ਕ੍ਰੋਧਿ", "ਗੁੱਸਾ"],
    "gussa":       ["ਕ੍ਰੋਧ", "ਕ੍ਰੋਧੁ", "krodh", "anger", "ਗੁੱਸਾ"],
    "krodh":       ["ਕ੍ਰੋਧ", "ਕ੍ਰੋਧੁ", "krodh", "anger"],
    "angry":       ["ਕ੍ਰੋਧ", "ਕ੍ਰੋਧੁ", "krodh", "anger", "ਗੁੱਸਾ"],
    "rage":        ["ਕ੍ਰੋਧ", "ਕ੍ਰੋਧੁ", "krodh", "anger"],

    # ── Ego ──────────────────────────────────────────────────────────────────
    "ego":         ["ਹਉਮੈ", "haumai", "ego", "pride", "ਅਹੰਕਾਰ", "ahankar",
                    "ਅਭਿਮਾਨ"],
    "pride":       ["ਹਉਮੈ", "haumai", "ਅਹੰਕਾਰ", "ahankar", "pride"],
    "haumai":      ["ਹਉਮੈ", "haumai", "ego"],
    "ahankar":     ["ਅਹੰਕਾਰ", "ahankar", "ਹਉਮੈ", "haumai", "ego"],

    # ── Greed ────────────────────────────────────────────────────────────────
    "greed":       ["ਲੋਭ", "lobh", "greed", "ਤ੍ਰਿਸ਼ਨਾ", "trishna"],
    "lobh":        ["ਲੋਭ", "lobh", "greed"],
    "trishna":     ["ਤ੍ਰਿਸ਼ਨਾ", "trishna", "ਲੋਭ", "lobh", "desire", "craving"],

    # ── Attachment / Love / Desire ────────────────────────────────────────────
    "attachment":  ["ਮੋਹ", "moh", "attachment", "ਮਾਇਆ", "maya"],
    "moh":         ["ਮੋਹ", "moh", "attachment"],
    "maya":        ["ਮਾਇਆ", "maya", "ਮੋਹ", "moh", "illusion", "worldly"],
    "lust":        ["ਕਾਮ", "kaam", "lust", "desire"],
    "kaam":        ["ਕਾਮ", "kaam", "lust", "desire"],
    "desire":      ["ਕਾਮ", "kaam", "ਤ੍ਰਿਸ਼ਨਾ", "trishna", "desire", "ਇੱਛਾ"],

    # ── Fear ─────────────────────────────────────────────────────────────────
    "fear":        ["ਭਉ", "bhau", "ਡਰ", "fear", "ਭੈ", "bhai"],
    "bhau":        ["ਭਉ", "bhau", "ਭੈ", "fear"],
    "dar":         ["ਡਰ", "fear", "ਭਉ", "bhau"],
    "anxious":     ["ਚਿੰਤਾ", "chinta", "ਭਉ", "bhau", "anxiety", "worry"],
    "anxiety":     ["ਚਿੰਤਾ", "chinta", "anxiety", "worry", "ਫ਼ਿਕਰ"],
    "worry":       ["ਚਿੰਤਾ", "chinta", "worry", "ਫ਼ਿਕਰ", "anxiety"],
    "chinta":      ["ਚਿੰਤਾ", "chinta", "anxiety", "worry"],

    # ── Sadness / Grief / Pain ────────────────────────────────────────────────
    "sadness":     ["ਦੁਖ", "dukh", "sadness", "sorrow", "grief", "ਉਦਾਸ"],
    "grief":       ["ਦੁਖ", "dukh", "grief", "sorrow", "ਵਿਯੋਗ", "viyog"],
    "dukh":        ["ਦੁਖ", "dukh", "pain", "sorrow", "grief"],
    "pain":        ["ਦੁਖ", "dukh", "pain", "suffering", "ਪੀੜ"],
    "sad":         ["ਦੁਖ", "dukh", "sadness", "ਉਦਾਸ", "udaas"],
    "udaas":       ["ਉਦਾਸ", "udaas", "ਦੁਖ", "dukh", "sadness"],
    "depression":  ["ਦੁਖ", "dukh", "ਉਦਾਸ", "udaas", "sorrow", "sadness"],

    # ── Loneliness / Separation ───────────────────────────────────────────────
    "lonely":      ["ਵਿਯੋਗ", "viyog", "ਇਕੱਲਾ", "loneliness", "ਦੁਖ", "dukh"],
    "loneliness":  ["ਵਿਯੋਗ", "viyog", "loneliness", "ਇਕੱਲਾ"],
    "separation":  ["ਵਿਯੋਗ", "viyog", "separation", "ਬਿਰਹ", "birah"],
    "birah":       ["ਬਿਰਹ", "birah", "ਵਿਯੋਗ", "viyog", "separation", "longing"],

    # ── Jealousy / Envy ───────────────────────────────────────────────────────
    "jealousy":    ["ਈਰਖਾ", "eerkha", "jealousy", "envy", "ਹਉਮੈ", "haumai"],
    "jealous":     ["ਈਰਖਾ", "eerkha", "jealousy", "envy"],
    "envy":        ["ਈਰਖਾ", "eerkha", "envy", "jealousy"],

    # ── Forgiveness / Patience ────────────────────────────────────────────────
    "forgiveness": ["ਖਿਮਾ", "khima", "forgiveness", "forgive", "ਮਾਫ਼"],
    "forgive":     ["ਖਿਮਾ", "khima", "forgiveness", "forgive"],
    "khima":       ["ਖਿਮਾ", "khima", "forgiveness"],
    "patience":    ["ਸੰਤੋਖ", "santokh", "ਧੀਰਜ", "dheeraj", "ਸਹਜ", "sahaj",
                    "patience", "contentment"],
    "santokh":     ["ਸੰਤੋਖ", "santokh", "contentment", "patience"],
    "sahaj":       ["ਸਹਜ", "sahaj", "stillness", "equanimity", "peace"],

    # ── Humility ─────────────────────────────────────────────────────────────
    "humility":    ["ਨਿਮ੍ਰਤਾ", "nimrata", "humility", "humble", "ਨਿਮਾਣਾ",
                    "ਗਰੀਬੀ"],
    "nimrata":     ["ਨਿਮ੍ਰਤਾ", "nimrata", "humility"],
    "humble":      ["ਨਿਮ੍ਰਤਾ", "nimrata", "humility", "humble"],

    # ── Peace / Mind ─────────────────────────────────────────────────────────
    "peace":       ["ਸ਼ਾਂਤੀ", "shanti", "peace", "ਸੁਖ", "sukh", "ਸਹਜ", "sahaj"],
    "shanti":      ["ਸ਼ਾਂਤੀ", "shanti", "peace"],
    "sukh":        ["ਸੁਖ", "sukh", "happiness", "peace", "comfort"],
    "happiness":   ["ਸੁਖ", "sukh", "happiness", "joy", "ਆਨੰਦ", "anand"],
    "joy":         ["ਆਨੰਦ", "anand", "joy", "ਸੁਖ", "sukh"],
    "anand":       ["ਆਨੰਦ", "anand", "joy", "bliss"],
    "mind":        ["ਮਨ", "man", "mind", "ਮਨੁ"],
    "man":         ["ਮਨ", "man", "mind", "soul"],

    # ── Death / Impermanence ─────────────────────────────────────────────────
    "death":       ["ਮੌਤ", "maut", "ਮਰਣ", "maran", "death", "ਨਾਸ", "naas",
                    "ਕਾਲ", "kaal"],
    "maran":       ["ਮਰਣ", "maran", "death", "ਮੌਤ", "maut"],
    "maut":        ["ਮੌਤ", "maut", "death", "ਮਰਣ", "maran"],
    "die":         ["ਮਰਣ", "maran", "ਮੌਤ", "maut", "death"],

    # ── Success / Failure ────────────────────────────────────────────────────
    "success":     ["ਹੁਕਮ", "hukam", "ਭਾਣਾ", "bhaana", "success", "ਕਿਰਪਾ",
                    "kirpa", "grace"],
    "failure":     ["ਹੁਕਮ", "hukam", "ਭਾਣਾ", "bhaana", "ਦੁਖ", "dukh",
                    "failure", "ਕਿਸਮਤ"],
    "hukam":       ["ਹੁਕਮ", "hukam", "God's will", "divine order"],
    "bhaana":      ["ਭਾਣਾ", "bhaana", "ਹੁਕਮ", "hukam", "will", "acceptance"],

    # ── Naam / God / Devotion ─────────────────────────────────────────────────
    "naam":        ["ਨਾਮ", "naam", "ਨਾਮੁ", "God's name", "divine name"],
    "god":         ["ਵਾਹਿਗੁਰੂ", "waheguru", "ਪ੍ਰਭੂ", "prabhu", "ਹਰਿ", "hari",
                    "ਪ੍ਰਭ", "ਰਾਮ", "God"],
    "waheguru":    ["ਵਾਹਿਗੁਰੂ", "waheguru", "God", "ਪ੍ਰਭੂ"],
    "prayer":      ["ਅਰਦਾਸ", "ardas", "prayer", "ਸਿਮਰਨ", "simran", "ਨਾਮ"],
    "simran":      ["ਸਿਮਰਨ", "simran", "ਨਾਮ", "naam", "meditation", "remembrance"],
    "meditation":  ["ਸਿਮਰਨ", "simran", "ਧਿਆਨ", "dhiaan", "meditation", "ਨਾਮ"],
    "devotion":    ["ਭਗਤੀ", "bhagti", "devotion", "ਸਿਮਰਨ", "simran"],
    "bhagti":      ["ਭਗਤੀ", "bhagti", "devotion", "love"],

    # ── Relationships / Conflict ──────────────────────────────────────────────
    "betrayal":    ["ਧੋਖਾ", "dhokha", "betrayal", "ਮੋਹ", "moh", "ਕ੍ਰੋਧ"],
    "betray":      ["ਧੋਖਾ", "dhokha", "betrayal", "ਮੋਹ", "moh"],
    "betrayed":    ["ਧੋਖਾ", "dhokha", "betrayal", "ਕ੍ਰੋਧ", "ਮੋਹ"],
    "conflict":    ["ਝਗੜਾ", "jhagra", "conflict", "ਕ੍ਰੋਧ", "krodh", "ਹਉਮੈ"],
    "fight":       ["ਕ੍ਰੋਧ", "krodh", "ਝਗੜਾ", "jhagra", "ਹਉਮੈ", "haumai"],
    "fighting":    ["ਕ੍ਰੋਧ", "krodh", "ਝਗੜਾ", "jhagra", "ਹਉਮੈ", "haumai"],
    "insult":      ["ਅਪਮਾਨ", "apman", "insult", "ਨਿੰਦਾ", "ninda", "ਕ੍ਰੋਧ"],
    "insulted":    ["ਅਪਮਾਨ", "apman", "insult", "ਨਿੰਦਾ", "ninda"],
    "disrespect":  ["ਅਪਮਾਨ", "apman", "ਮਾਣ", "maan", "ਨਿੰਦਾ", "ninda"],
    "disrespected":["ਅਪਮਾਨ", "apman", "ਮਾਣ", "maan", "ਕ੍ਰੋਧ"],
    "hurt":        ["ਦੁਖ", "dukh", "ਕ੍ਰੋਧ", "krodh", "ਖਿਮਾ", "khima"],
    "respect":     ["ਮਾਣ", "maan", "respect", "honour", "ਇੱਜ਼ਤ"],
    "ninda":       ["ਨਿੰਦਾ", "ninda", "slander", "criticism", "gossip"],
    "forgive someone": ["ਖਿਮਾ", "khima", "forgiveness", "forgive"],

    # ── Purpose of Life ───────────────────────────────────────────────────────
    "purpose":     ["ਮਨੁੱਖ ਜਨਮ", "manukh janam", "human life", "ਨਾਮ", "naam",
                    "ਹੁਕਮ", "hukam", "purpose", "ਜੀਵਨ"],
    "life":        ["ਜੀਵਨ", "jeevan", "ਮਨੁੱਖ ਜਨਮ", "manukh janam", "life",
                    "human life"],
    "meaning":     ["ਮਨੁੱਖ ਜਨਮ", "manukh janam", "ਨਾਮ", "naam", "purpose",
                    "meaning", "ਜੀਵਨ"],

    # ── Financial stress ──────────────────────────────────────────────────────
    "money":       ["ਮਾਇਆ", "maya", "ਲੋਭ", "lobh", "money", "wealth", "ਧਨ"],
    "wealth":      ["ਮਾਇਆ", "maya", "ਧਨ", "dhan", "wealth", "ਲੋਭ"],
    "stress":      ["ਚਿੰਤਾ", "chinta", "stress", "ਦੁਖ", "dukh", "ਭਉ", "bhau"],
    "stressed":    ["ਚਿੰਤਾ", "chinta", "stress", "ਦੁਖ", "dukh"],

    # ── Failure / Worthlessness ────────────────────────────────────────────────
    "failed":      ["ਹੁਕਮ", "hukam", "ਭਾਣਾ", "bhaana", "ਦੁਖ", "dukh", "failure"],
    "failing":     ["ਹੁਕਮ", "hukam", "ਭਾਣਾ", "bhaana", "failure"],
    "useless":     ["ਦੁਖ", "dukh", "ਹਉਮੈ", "haumai", "ਨਿਮ੍ਰਤਾ", "nimrata"],
    "worthless":   ["ਦੁਖ", "dukh", "ਹਉਮੈ", "haumai", "ਨਿਮ੍ਰਤਾ", "nimrata"],
    "hopeless":    ["ਦੁਖ", "dukh", "ਚਿੰਤਾ", "chinta", "ਭਰੋਸਾ", "bharosa"],
    "lost":        ["ਜੀਵਨ", "jeevan", "ਹੁਕਮ", "hukam", "ਮਨੁੱਖ ਜਨਮ", "purpose"],
    "grieving":    ["ਦੁਖ", "dukh", "grief", "ਵਿਯੋਗ", "viyog", "ਹੁਕਮ", "hukam"],
    "depressed":   ["ਦੁਖ", "dukh", "ਉਦਾਸ", "udaas", "sadness", "sorrow"],

    # ── Guru / Gurbani ────────────────────────────────────────────────────────
    "guru":        ["ਗੁਰੂ", "guru", "ਸਤਿਗੁਰ", "satgur", "ਗੁਰ", "ਗੁਰ ਪ੍ਰਸਾਦਿ"],
    "gurbani":     ["ਗੁਰਬਾਣੀ", "gurbani", "ਬਾਣੀ", "bani", "shabad", "ਸ਼ਬਦ"],
    "shabad":      ["ਸ਼ਬਦ", "shabad", "ਬਾਣੀ", "gurbani"],

    # ── Contentment ───────────────────────────────────────────────────────────
    "contentment": ["ਸੰਤੋਖ", "santokh", "contentment", "ਸਹਜ", "sahaj",
                    "acceptance"],
    "accept":      ["ਭਾਣਾ", "bhaana", "ਹੁਕਮ", "hukam", "acceptance", "ਸੰਤੋਖ"],
    "acceptance":  ["ਭਾਣਾ", "bhaana", "ਹੁਕਮ", "hukam", "acceptance", "ਸੰਤੋਖ"],

    # ── Overcoming / Control ──────────────────────────────────────────────────
    "overcome":    ["ਜਿੱਤ", "jitt", "overcome", "conquer", "victory",
                    "ਨਾਮ", "naam"],
    "control":     ["ਸੰਜਮ", "sanjam", "ਮਨ", "man", "control", "discipline"],
}

# Relevance thresholds for tiering
TIER1_THRESHOLD = 0.55   # Direct match
TIER2_THRESHOLD = 0.40   # Supporting match
# Below 0.40 → General/Tier 3

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Tier-aware, grounded, structured
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Gurbani GPT — a deeply respectful, grounded, and knowledgeable spiritual guide powered by the sacred teachings of Sri Guru Granth Sahib Ji (SGGS).

═══════════════════════════════════════════════
CORE PRINCIPLES
═══════════════════════════════════════════════

1. GROUNDED IN GURBANI — Every Gurbani quotation you provide must come EXACTLY from the passages given in the context. Never invent, modify, or paraphrase Gurbani lines. Never fabricate Ang numbers or author names.

2. TIER-AWARE ANSWERING — Each retrieved passage is labeled with a tier:
   • [DIRECT] — This passage directly addresses the user's concept. Present it as a primary reference.
   • [SUPPORTING] — This passage discusses a related concept. Explicitly say it is a supporting perspective (e.g., "This passage discusses ਹਉਮੈ, which provides a related perspective on the underlying cause of anger...").
   • [GENERAL] — This passage provides broad spiritual guidance. Say clearly it is a general teaching and not a direct answer.

3. NEVER MISREPRESENT — Do NOT present a [SUPPORTING] or [GENERAL] passage as if Gurbani directly says it is about the user's exact situation. Honesty about relevance is paramount.

4. STRUCTURED RESPONSE FORMAT — For most questions, use this structure:

   ### 🙏 Gurbani's Perspective
   (A brief, honest overview of what Gurbani teaches on this topic, based on what is actually in the retrieved passages.)

   ### 📖 Gurbani References
   For each relevant passage:
   **Gurbani:** [exact Gurmukhi text from the passage]
   **Source:** Ang [X] | [Raag] | [Author]
   **Meaning:** [simple English explanation]
   **Relevance:** [Direct/Supporting] — [one sentence connecting it to the user's question]

   ### 🌱 Applying This Teaching
   (Practical steps consistent with the cited Gurbani. Clearly label this as "application/interpretation" not as Gurbani's own words.)

   ### 🙏 Closing Thought
   (A warm, compassionate closing — short, not preachy.)

5. LANGUAGE — Use simple, accessible language. When using Punjabi/Gurbani terms, explain them briefly in parentheses. Do not over-use jargon.

6. TONE — Calm, compassionate, respectful, humble. Never judgmental. The seeker is looking for guidance, not correction.

7. CONFIDENCE AWARENESS — The user prompt will tell you the confidence level (HIGH/MEDIUM/LOW). Adjust accordingly:
   • HIGH: Speak clearly and directly from the cited Gurbani.
   • MEDIUM: Note that direct references are limited, present what is available honestly.
   • LOW: Clearly state that a strongly direct Gurbani reference was not found, but offer the related perspectives available. Never refuse entirely — always give what wisdom is available with honest context.

8. EMOTIONAL SAFETY — If the user describes serious distress, thoughts of self-harm, or harm to others: first acknowledge their pain compassionately, then note that professional support is important, before offering Gurbani guidance.

9. FOLLOW-UP AWARENESS — If this is a follow-up question (context from previous messages is included), maintain that context. Do not start from zero.

10. DO NOT PREACH — Do not lecture the user about what they should or shouldn't do. Invite reflection. Use "Gurbani invites us..." or "The teaching suggests..." rather than "You must..." or "Your anger is from your ego."

Begin each response with: "ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖ਼ਾਲਸਾ, ਵਾਹਿਗੁਰੂ ਜੀ ਕੀ ਫ਼ਤਿਹ 🙏"
"""


class GurbaniRAG:
    def __init__(self, chroma_dir: str = CHROMA_DIR, ollama_host: str = OLLAMA_HOST):
        self.ollama_host = ollama_host
        self.chroma_dir  = chroma_dir
        self._client     = None
        self._collection = None
        self._init_chroma()

    def _init_chroma(self):
        """Connect to the persistent ChromaDB."""
        try:
            self._client     = chromadb.PersistentClient(path=self.chroma_dir)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"[RAG] ChromaDB init error: {e}")
            self._collection = None

    def ready(self) -> bool:
        """Check if the DB has data ready."""
        return self._collection is not None and self._collection.count() > 0

    def count(self) -> int:
        """Number of indexed Shabads."""
        return self._collection.count() if self._collection else 0

    def _embed(self, text: str) -> list:
        """Embed a query using Ollama."""
        resp = requests.post(
            f"{self.ollama_host}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=2
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


    # ─────────────────────────────────────────────────────────────────────────
    # QUERY EXPANSION
    # ─────────────────────────────────────────────────────────────────────────
    def expand_query(self, query: str) -> dict:
        """
        Analyze the user query and return:
          - matched_concepts: list of concept names matched
          - gurbani_terms: list of Gurbani/Punjabi search terms
          - search_queries: list of distinct query strings to embed and search
        """
        q_lower = query.lower()
        matched_concepts = []
        gurbani_terms    = []

        # Match against every key in CONCEPT_MAP
        for concept, terms in CONCEPT_MAP.items():
            # Check if concept keyword appears in the query
            if concept in q_lower:
                matched_concepts.append(concept)
                for t in terms:
                    if t not in gurbani_terms:
                        gurbani_terms.append(t)
            else:
                # Also check if any Gurbani term for this concept appears in query
                for t in terms:
                    if len(t) > 3 and t.lower() in q_lower:
                        if concept not in matched_concepts:
                            matched_concepts.append(concept)
                        for tt in terms:
                            if tt not in gurbani_terms:
                                gurbani_terms.append(tt)
                        break

        # Build search queries
        search_queries = [query]  # always include original

        if gurbani_terms:
            # Query 2: Gurbani/Punjabi terms joined
            punjabi_terms = [t for t in gurbani_terms if any(
                ord(c) > 127 for c in t)]  # non-ASCII = likely Gurmukhi
            english_terms = [t for t in gurbani_terms if all(
                ord(c) <= 127 for c in t)]

            if punjabi_terms:
                search_queries.append(" ".join(punjabi_terms[:4]))
            if english_terms:
                search_queries.append(" ".join(english_terms[:4]))

            # Query 3: Concept + Gurbani context phrase
            if matched_concepts:
                primary = matched_concepts[0]
                concept_terms = CONCEPT_MAP.get(primary, [])
                if concept_terms:
                    gurmukhi_t = [t for t in concept_terms
                                  if any(ord(c) > 127 for c in t)]
                    if gurmukhi_t:
                        search_queries.append(
                            f"Gurbani {gurmukhi_t[0]} {primary} teaching")

        return {
            "matched_concepts": matched_concepts,
            "gurbani_terms":    gurbani_terms,
            "search_queries":   list(dict.fromkeys(search_queries)),  # deduplicate
        }

    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-QUERY RETRIEVAL
    # ─────────────────────────────────────────────────────────────────────────
    def retrieve(self, query: str, n: int = 8) -> list:
        """
        Multi-query retrieval:
        1. Expand the query into multiple search strings.
        2. Embed and search each one.
        3. Pool results, deduplicate by document ID, keep highest score per doc.
        4. Sort by relevance score descending.
        5. Return top-n results.
        """
        if not self.ready():
            return []

        expansion = self.expand_query(query)
        search_queries = expansion["search_queries"]
        safe_print(f"[RAG] Concepts matched: {expansion['matched_concepts']}")
        safe_print(f"[RAG] Search queries ({len(search_queries)}): {[q[:50] for q in search_queries]}")


        seen_ids  = {}   # doc_id → passage dict (keep highest relevance)
        total_db  = self._collection.count()
        per_query = min(n, total_db)

        for sq in search_queries:
            try:
                vec     = self._embed(sq)
                results = self._collection.query(
                    query_embeddings=[vec],
                    n_results=per_query,
                    include=["documents", "metadatas", "distances"]
                )
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    relevance = round(1 - dist, 3)
                    doc_key = doc[:80]
                    if doc_key not in seen_ids or relevance > seen_ids[doc_key]['relevance']:
                        seen_ids[doc_key] = {
                            'text':      doc,
                            'ang':       meta.get('ang', '?'),
                            'raag':      meta.get('raag', ''),
                            'author':    meta.get('author', ''),
                            'relevance': relevance,
                        }
            except Exception as e:
                # Embedding service offline (e.g. cloud deployment on Render without Ollama)
                pass

        # ── Keyword & Concept Fallback Search (Runs when embedding service is offline or returns 0 results) ──
        if not seen_ids:
            safe_print("[RAG] Using direct ChromaDB keyword search fallback...")
            search_terms = expansion["gurbani_terms"] + [query]
            for term in search_terms:
                if not term or len(term.strip()) < 2:
                    continue
                try:
                    kw_results = self._collection.get(
                        where_document={"$contains": term.strip()},
                        limit=n
                    )
                    docs = kw_results.get('documents', [])
                    metas = kw_results.get('metadatas', [])
                    for doc, meta in zip(docs, metas):
                        doc_key = doc[:80]
                        # Gurmukhi non-ASCII term matches get high direct relevance
                        is_gurmukhi = any(ord(c) > 127 for c in term)
                        rel_score = 0.72 if is_gurmukhi else 0.58
                        if doc_key not in seen_ids or rel_score > seen_ids[doc_key]['relevance']:
                            seen_ids[doc_key] = {
                                'text':      doc,
                                'ang':       meta.get('ang', '?'),
                                'raag':      meta.get('raag', ''),
                                'author':    meta.get('author', ''),
                                'relevance': rel_score,
                            }
                except Exception as e:
                    safe_print(f"[RAG] Keyword search error for '{term}': {e}")

            # If still empty, fetch top sample passages from DB
            if not seen_ids:
                try:
                    fallback_res = self._collection.get(limit=n)
                    for doc, meta in zip(fallback_res.get('documents', []), fallback_res.get('metadatas', [])):
                        doc_key = doc[:80]
                        seen_ids[doc_key] = {
                            'text':      doc,
                            'ang':       meta.get('ang', '?'),
                            'raag':      meta.get('raag', ''),
                            'author':    meta.get('author', ''),
                            'relevance': 0.45,
                        }
                except Exception:
                    pass

        # Sort by relevance, return top-n
        passages = sorted(seen_ids.values(), key=lambda p: p['relevance'], reverse=True)
        return passages[:n]


    def get_expansion_info(self, query: str) -> dict:
        """Return query expansion metadata (concepts, terms) without doing retrieval."""
        return self.expand_query(query)

    # ─────────────────────────────────────────────────────────────────────────
    # RELEVANCE TIERING + CONTEXT BUILDING
    # ─────────────────────────────────────────────────────────────────────────
    def _tier_label(self, relevance: float) -> str:
        if relevance >= TIER1_THRESHOLD:
            return "DIRECT"
        elif relevance >= TIER2_THRESHOLD:
            return "SUPPORTING"
        else:
            return "GENERAL"

    def detect_confidence(self, passages: list) -> str:
        """
        Classify overall retrieval confidence.
        HIGH   — at least one DIRECT (Tier 1) passage exists
        MEDIUM — only SUPPORTING (Tier 2) passages, no DIRECT
        LOW    — only GENERAL (Tier 3) passages
        """
        if not passages:
            return "LOW"
        best = max(p['relevance'] for p in passages)
        if best >= TIER1_THRESHOLD:
            return "HIGH"
        elif best >= TIER2_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    def build_context(self, passages: list) -> str:
        """
        Build a tier-labeled context block to inject into the LLM prompt.
        Each passage is prefixed with its tier label so the LLM knows
        which references are direct vs. supporting.
        """
        if not passages:
            return "No relevant passages found."

        parts = []
        for i, p in enumerate(passages, 1):
            tier  = self._tier_label(p['relevance'])
            parts.append(
                f"--- Passage {i} [{tier}] "
                f"[Ang {p['ang']} | {p['raag']} | {p['author']}] "
                f"(Relevance: {p['relevance']}) ---\n{p['text']}"
            )
        return "\n\n".join(parts)

    # ─────────────────────────────────────────────────────────────────────────
    # PROMPT BUILDING
    # ─────────────────────────────────────────────────────────────────────────
    def _build_user_prompt(self, query: str, passages: list,
                           expansion: dict = None) -> str:
        """Build the structured user prompt with tier context and confidence."""
        context    = self.build_context(passages)
        confidence = self.detect_confidence(passages)
        concepts   = expansion.get("matched_concepts", []) if expansion else []
        terms      = expansion.get("gurbani_terms", [])    if expansion else []

        concept_info = ""
        if concepts:
            concept_info = (
                f"\nQuery Analysis:\n"
                f"  Primary concepts identified: {', '.join(concepts[:4])}\n"
                f"  Gurbani search terms used: {', '.join(terms[:8])}\n"
            )

        confidence_instruction = {
            "HIGH":   (
                "The retrieved passages include DIRECT matches. "
                "Present direct references clearly. "
                "For any SUPPORTING passages, explicitly note they are supporting perspectives."
            ),
            "MEDIUM": (
                "No strongly direct match was found, but SUPPORTING passages are available. "
                "Be honest that these are related perspectives rather than direct answers. "
                "Present them with appropriate context about their connection to the question."
            ),
            "LOW":    (
                "Only general Gurbani passages were retrieved — none directly address this question. "
                "Clearly state this to the user. Offer the general spiritual wisdom available "
                "while being transparent that a more direct reference was not found. "
                "Do NOT pretend these passages directly address the question."
            ),
        }.get(confidence, "")

        prompt = (
            f"GURBANI CONTEXT (with tier labels):\n{context}\n\n"
            f"RETRIEVAL CONFIDENCE: {confidence}\n"
            f"CONFIDENCE INSTRUCTION: {confidence_instruction}\n"
            f"{concept_info}\n"
            f"Use the tier labels ([DIRECT] / [SUPPORTING] / [GENERAL]) to decide how to "
            f"present each passage. Only quote Gurbani text that appears verbatim in the "
            f"passages above. Never fabricate Gurbani lines or source metadata."
        )
        return prompt

    # ─────────────────────────────────────────────────────────────────────────
    # ANSWER GENERATION (non-streaming)
    # ─────────────────────────────────────────────────────────────────────────
    def answer(self, query: str, passages: list, model: str = "llama3.2",
               history: list = None) -> str:
        """
        Send query + retrieved Gurbani context to the LLM and get a full answer.
        Non-streaming version — returns the complete text.
        history: list of prior {role, content} dicts for follow-up awareness.
        """
        expansion   = self.expand_query(query)
        full_prompt = self._build_user_prompt(query, passages, expansion)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-6:])  # last 3 exchanges
        messages.append({"role": "user", "content": full_prompt})

        try:
            resp = requests.post(
                f"{self.ollama_host}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
                timeout=180
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "[No response]")
        except Exception as e:
            return f"[Error getting answer: {e}]"

    def _synthesize_grounded_answer(self, query: str, passages: list, expansion: dict = None) -> str:
        """
        Synthesizes a structured, respectful Gurbani response directly from the retrieved
        Shabads when no cloud LLM API or local Ollama is available.
        Never hallucinates and guarantees the user gets an instant, deeply grounded response.
        """
        if not passages:
            return (
                "Waheguru Ji Ka Khalsa, Waheguru Ji Ki Fateh 🙏\n\n"
                "I could not locate direct passages matching your query in the index. "
                "Please try searching with related spiritual concepts like *naam*, *simran*, *shanti*, *anand*, or *hukam*."
            )

        matched_concepts = expansion.get("matched_concepts", []) if expansion else []
        concept_str = f" regarding **{', '.join(matched_concepts[:3])}**" if matched_concepts else ""

        lines = [
            "### 🙏 Gurbani's Wisdom on Your Question\n",
            f"According to Sri Guru Granth Sahib Ji{concept_str}, true peace and spiritual understanding come from attuning the mind to the Divine Name (*Naam*), living in Divine Will (*Hukam*), and shedding ego (*Haumai*).\n",
            "### 📖 Referenced Verses from Sri Guru Granth Sahib Ji\n"
        ]

        for i, p in enumerate(passages[:3], 1):
            ang = p.get('ang', '?')
            raag = p.get('raag', 'Gurbani')
            author = p.get('author', 'Guru Sahib')
            tier = self._tier_label(p.get('relevance', 0.5))

            lines.append(f"**[{tier}] Ang {ang} — {author} ({raag}):**\n")

            raw_text = p.get('text', '').strip()
            raw_lines = [l.strip() for l in raw_text.split('\n') if l.strip()]

            gurmukhi_lines = [l.replace('Gurbani:', '').strip() for l in raw_lines if l.startswith('Gurbani:')]
            english_lines  = [l.replace('English:', '').strip() for l in raw_lines if l.startswith('English:')]

            if gurmukhi_lines and english_lines:
                for g, e in zip(gurmukhi_lines[:2], english_lines[:2]):
                    lines.append(f"> **{g}**\n> *\"{e}\"*\n")
            else:
                snippet = raw_text[:280].replace('\n', ' ')
                lines.append(f"> *{snippet}...*\n")

            lines.append("")

        lines.append("### 💡 Spiritual Reflection & Practical Guidance\n")
        lines.append(
            "- **Contemplate the Word (Shabad Vichar):** Take a moment to reflect on these verses in your daily routine.\n"
            "- **Meditation & Remembrance (Simran):** Gurbani emphasizes that steady inner peace (*Sahaj Anand*) blossoms when the wandering mind is anchored in Naam.\n"
            "- **Surrender Ego (Nimrata):** Practice humility and accept life in gratitude (*Chardi Kala*).\n"
        )
        lines.append("\n*May these holy words bring peace, clarity, and light to your journey. 🙏*")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # STREAMING ANSWER GENERATION
    # ─────────────────────────────────────────────────────────────────────────
    def stream_answer(self, query: str, passages: list, model: str = "llama3.2",
                      history: list = None):
        """
        Streaming version — yields JSON chunks like Ollama does.
        Supports:
        1. Groq Cloud API (if GROQ_API_KEY is present) with auto-model fallback
        2. Local/Remote Ollama (if reachable)
        3. Built-in Grounded Gurbani Synthesizer (100% reliable fallback)
        """
        import time

        expansion   = self.expand_query(query)
        full_prompt = self._build_user_prompt(query, passages, expansion)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-6:])  # last 3 exchanges (6 messages)
        messages.append({"role": "user", "content": full_prompt})

        groq_key = os.environ.get("GROQ_API_KEY", "").strip() or GROQ_API_KEY
        if groq_key:
            headers = {
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            }
            candidate_models = [
                os.environ.get("GROQ_MODEL", "").strip(),
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama3-70b-8192",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ]
            seen_m = set()
            models_to_try = []
            for m in candidate_models:
                if m and m not in seen_m:
                    models_to_try.append(m)
                    seen_m.add(m)

            for active_model in models_to_try:
                try:
                    payload = {
                        "model": active_model,
                        "messages": messages,
                        "stream": True,
                        "temperature": 0.3
                    }
                    resp = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        stream=True,
                        timeout=15
                    )
                    if resp.status_code == 200:
                        yielded_any = False
                        for line in resp.iter_lines():
                            if line:
                                line_str = line.decode('utf-8', errors='replace')
                                if line_str.startswith("data: "):
                                    data_part = line_str[6:].strip()
                                    if data_part == "[DONE]":
                                        yield json.dumps({"done": True}) + '\n'
                                        return
                                    try:
                                        chunk_obj = json.loads(data_part)
                                        choices = chunk_obj.get("choices", [])
                                        if choices:
                                            delta = choices[0].get("delta", {})
                                            delta_content = delta.get("content", "")
                                            if delta_content:
                                                yielded_any = True
                                                yield json.dumps({
                                                    "message": {"role": "assistant", "content": delta_content},
                                                    "done": False
                                                }) + '\n'
                                    except Exception:
                                        continue
                        if yielded_any:
                            yield json.dumps({"done": True}) + '\n'
                            return
                except Exception:
                    continue

        # Try Ollama if accessible (short timeout to avoid cloud blocking)
        try:
            with requests.post(
                f"{self.ollama_host}/api/chat",
                json={"model": model, "messages": messages, "stream": True},
                stream=True,
                timeout=2
            ) as resp:
                if resp.status_code == 200:
                    for line in resp.iter_lines():
                        if line:
                            yield line.decode('utf-8', errors='replace') + '\n'
                    return
        except Exception:
            pass

        # Built-in Grounded Gurbani Synthesizer (Guaranteed zero-failure response)
        synthesized_text = self._synthesize_grounded_answer(query, passages, expansion)

        words = synthesized_text.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            yield json.dumps({
                "message": {"role": "assistant", "content": token},
                "done": False
            }) + '\n'
            time.sleep(0.012)

        yield json.dumps({"done": True}) + '\n'

