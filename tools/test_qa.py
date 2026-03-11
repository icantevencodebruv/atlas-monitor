#!/usr/bin/env python3
"""
QA model audit — two-part test.

Usage:
    python tools/test_qa.py

Run from the project root. Writes output to tools/qa_test_results.txt.
Requires the model to be downloaded first:
    bash tools/download_qa_model.sh
"""

import os
import sqlite3
import sys

# Run from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa_test_results.txt")

LONG_PARAGRAPH = (
    "So basically what happened was that we were looking at the architecture and we realized "
    "that the current approach is not really scalable. I mean we have been talking about this "
    "for a while now and the consensus is that we need to refactor the whole data layer. "
    "Leon suggested we start with the database abstraction and Hugo agreed that makes sense "
    "given the current technical debt. The timeline is probably two sprints if everything goes "
    "according to plan but realistically we should buffer for at least one extra sprint because "
    "there are always unexpected complications when you are touching the core infrastructure."
)

CASES = [
    ("Hugo", "Thank you for watching!", "filtered"),
    ("Hugo", "Subtitles by the Amara community", "filtered"),
    ("Leon", "Thanks for watching, subscribe for more", "filtered"),
    ("Hugo", "", "filtered"),
    ("Hugo", " ", "filtered"),
    ("Hugo", "hmm", "pass"),    # valid filler — not filtered by regex or QA
    ("Hugo", "uh huh", "pass"),  # valid filler — not filtered by regex or QA
    ("Leon", "[music]", "filtered"),
    ("Hugo", "Ich glaube wir sollten das morgen besprechen", "pass"),
    ("Leon", "the the the project is is ready", "pass"),
    ("Hugo", "I think we should we should probably start with the the backend", "pass"),
    ("Leon", "ja genau also ich hab das gemacht", "pass"),
    ("Hugo", "and then I was like äh also basically we need to refactor this", "pass"),
    ("Leon", "Tizian hat gesagt er kommt um drei", "pass"),
    ("Hugo", "ja", "pass"),
    ("Hugo", LONG_PARAGRAPH, "pass"),
]


def run():
    from app.config import load_config
    config = load_config(os.environ.get("APP_CONFIG", "config.yaml"))
    from app.services.transcript_qa import TranscriptQA
    from app.services.transcription import is_non_speech
    qa = TranscriptQA(config.llm_qa.model_dump())

    def evaluate_full_pipeline(speaker: str, text: str):
        """Mirror the real pipeline: regex gate → QA model."""
        if is_non_speech(text):
            from app.services.transcript_qa import QAResult
            return QAResult(original_text=text, corrected_text="", was_modified=True, action="filtered")
        return qa.evaluate(speaker, text)

    lines = []

    def log(msg=""):
        lines.append(msg)
        print(msg)

    log("=" * 72)
    log("PART A — hardcoded edge cases")
    log("=" * 72)

    if not qa.model_loaded:
        log("WARNING: model not loaded — all results will be stub pass-throughs")

    mismatches = 0
    for i, (speaker, text, expected) in enumerate(CASES, 1):
        result = evaluate_full_pipeline(speaker, text)
        match = result.action == expected
        if not match:
            mismatches += 1
        status = "OK" if match else "MISMATCH"
        short_text = (text[:60] + "...") if len(text) > 60 else text
        log(
            f"[{i:02d}] {status:8s}  speaker={speaker}  expected={expected:9s}  "
            f"got={result.action:9s}  text={repr(short_text)}"
        )
        if result.was_modified:
            log(f"          corrected → {repr(result.corrected_text[:80])}")

    log()
    log(f"Part A summary: {len(CASES)} cases, {mismatches} mismatches")

    log()
    log("=" * 72)
    log("PART B — DB audit (last 100 transcripts)")
    log("=" * 72)

    db_path = config.storage.db_path
    if not os.path.exists(db_path):
        log(f"DB not found at {db_path} — skipping Part B")
    else:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, speaker, text, original_text FROM transcripts ORDER BY id DESC LIMIT 100"
        ).fetchall()
        conn.close()

        log(f"Rows fetched: {len(rows)}")
        log()

        total = len(rows)
        counts = {"pass": 0, "corrected": 0, "filtered": 0}
        flagged = 0

        for row in rows:
            speaker = row["speaker"]
            source_text = row["original_text"] or row["text"]
            result = evaluate_full_pipeline(speaker, source_text)
            counts[result.action] += 1

            flags = []
            if result.corrected_text and len(result.corrected_text) > len(source_text) * 1.2:
                flags.append("HALLUCINATION RISK")

            if source_text:
                original_proper = {w for w in source_text.split() if w and w[0].isupper() and w.isalpha()}
                corrected_proper = {w for w in result.corrected_text.split() if w and w[0].isupper() and w.isalpha()}
                lost = original_proper - corrected_proper
                if lost:
                    flags.append(f"PROPER NOUN LOST: {', '.join(sorted(lost))}")

            if flags:
                flagged += 1

            short_orig = (source_text[:50] + "...") if len(source_text) > 50 else source_text
            short_corr = (result.corrected_text[:50] + "...") if len(result.corrected_text) > 50 else result.corrected_text
            flag_str = "  *** " + " | ".join(flags) if flags else ""
            log(
                f"[{row['id']:5d}] {speaker:10s}  {result.action:9s}  "
                f"orig={repr(short_orig)}  corr={repr(short_corr)}{flag_str}"
            )

        log()
        log(
            f"Part B summary: total={total}  pass={counts['pass']}  "
            f"corrected={counts['corrected']}  filtered={counts['filtered']}  "
            f"flagged={flagged}"
        )

    log()
    log("Done.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults written to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
