#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 2 — Clean & Chunk Gurbani Data into Shabads
Reads raw JSON files (GurbaniNow API v2 format), groups lines by Shabad,
and creates clean chunks ready for embedding.
Run: python scripts/02_clean_chunk.py
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import os
import ast

RAW_DIR       = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)


def safe_parse(val, key=None):
    """Safely extract a value that may be a dict-string or plain string."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('{'):
            try:
                return ast.literal_eval(val)
            except Exception:
                pass
    return {}


def extract_lines_from_ang(ang_data):
    """Pull all lines from an Ang JSON response (new GurbaniNow format)."""
    lines = []
    page = ang_data.get('page', [])
    ang_no = int(ang_data.get('pageno', 0))

    for item in page:
        line = item.get('line', {})
        if not line:
            continue

        # Parse nested fields (they come as dict-strings in some cases)
        gurmukhi_raw   = safe_parse(line.get('gurmukhi', {}))
        transl_raw     = safe_parse(line.get('transliteration', {}))
        translation_raw = safe_parse(line.get('translation', {}))
        writer_raw     = safe_parse(line.get('writer', {}))
        raag_raw       = safe_parse(line.get('raag', {}))

        # Extract Gurmukhi unicode text
        gurmukhi = gurmukhi_raw.get('unicode', '') if isinstance(gurmukhi_raw, dict) else ''
        if not gurmukhi:
            continue

        # English transliteration
        transl_en = ''
        if isinstance(transl_raw, dict):
            en = transl_raw.get('english', {})
            transl_en = en.get('text', '') if isinstance(en, dict) else str(en)

        # English translation
        eng_text = ''
        if isinstance(translation_raw, dict):
            eng = translation_raw.get('english', {})
            if isinstance(eng, dict):
                eng_text = eng.get('default', '')
                if isinstance(eng_text, dict):
                    eng_text = eng_text.get('text', '')
            elif isinstance(eng, str):
                eng_text = eng

        # Writer
        writer_en  = writer_raw.get('english', '') if isinstance(writer_raw, dict) else ''
        writer_gur = writer_raw.get('unicode', '') if isinstance(writer_raw, dict) else ''

        # Raag
        raag_en  = raag_raw.get('english', '') if isinstance(raag_raw, dict) else ''
        raag_gur = raag_raw.get('unicode', '') if isinstance(raag_raw, dict) else ''

        lines.append({
            'shabad_id':       line.get('shabadid', ''),
            'ang':             ang_no,
            'line_no':         line.get('lineno', 0),
            'gurmukhi':        gurmukhi,
            'romanized':       transl_en,
            'english':         eng_text,
            'writer_english':  writer_en,
            'writer_gurmukhi': writer_gur,
            'raag_english':    raag_en,
            'raag_gurmukhi':   raag_gur,
        })

    return lines


def group_into_shabads(all_lines):
    """Group individual lines by Shabad ID to form complete chunks."""
    shabads = {}
    for line in all_lines:
        sid = line['shabad_id']
        if not sid:
            continue
        if sid not in shabads:
            shabads[sid] = {
                'shabad_id':       sid,
                'ang_start':       line['ang'],
                'raag_english':    line['raag_english'],
                'raag_gurmukhi':   line['raag_gurmukhi'],
                'writer_english':  line['writer_english'],
                'writer_gurmukhi': line['writer_gurmukhi'],
                'lines':           []
            }
        shabads[sid]['lines'].append(line)
        if line['ang'] > shabads[sid]['ang_start']:
            shabads[sid].setdefault('ang_end', line['ang'])
        else:
            shabads[sid].setdefault('ang_end', line['ang'])

    return list(shabads.values())


def build_chunk_text(shabad):
    """Build a single text block for a Shabad — this is what gets embedded."""
    lines  = shabad['lines']
    ang    = shabad['ang_start']
    raag   = shabad['raag_english']
    auth   = shabad['writer_english']

    parts = [f"[Ang {ang} | Raag: {raag} | Author: {auth}]"]

    for l in lines:
        if l.get('gurmukhi'):
            parts.append(f"Gurbani: {l['gurmukhi']}")
        if l.get('romanized'):
            parts.append(f"Transliteration: {l['romanized']}")
        if l.get('english'):
            parts.append(f"English: {l['english']}")
        parts.append("")  # blank line between lines

    return "\n".join(parts).strip()


def main():
    print()
    print("=" * 55)
    print("   Gurbani GPT - Chunking Shabads...")
    print("=" * 55)
    print()

    raw_files = sorted([
        f for f in os.listdir(RAW_DIR) if f.endswith('.json')
    ])

    if not raw_files:
        print("  ERROR: No raw data found! Run 01_download_data.py first.")
        return

    print(f"  Found {len(raw_files)} Ang files to process...")
    print()

    all_lines = []
    errors    = 0
    for i, fname in enumerate(raw_files):
        path = os.path.join(RAW_DIR, fname)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            ang_lines = extract_lines_from_ang(data)
            all_lines.extend(ang_lines)
        except Exception as e:
            errors += 1
            print(f"  WARN: Could not parse {fname}: {e}")
        if (i+1) % 200 == 0:
            print(f"  Processed {i+1}/{len(raw_files)} Angs -- {len(all_lines)} lines so far")

    print(f"\n  Total lines extracted: {len(all_lines)}")
    if errors:
        print(f"  Parse errors: {errors}")

    if not all_lines:
        print("\n  ERROR: No lines extracted. Check raw data format.")
        return

    # Group into Shabads
    shabads = group_into_shabads(all_lines)
    print(f"  Total Shabads (chunks): {len(shabads)}")

    # Build final chunk documents
    chunks = []
    for shabad in shabads:
        text = build_chunk_text(shabad)
        if len(text.strip()) < 20:
            continue
        chunks.append({
            'id':             shabad['shabad_id'],
            'ang':            shabad['ang_start'],
            'raag_english':   shabad['raag_english'],
            'writer_english': shabad['writer_english'],
            'text':           text,
            'line_count':     len(shabad['lines']),
        })

    # Save
    out_path = os.path.join(PROCESSED_DIR, 'gurbani_chunks.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\n  Saved {len(chunks)} chunks -> {out_path}")
    print()

    # Show a sample
    if chunks:
        print("  Sample Chunk:")
        print("  " + "-" * 50)
        sample = chunks[0]['text'][:400].replace('\n', '\n  ')
        print("  " + sample)
        print("  " + "-" * 50)
    print()


if __name__ == '__main__':
    main()
