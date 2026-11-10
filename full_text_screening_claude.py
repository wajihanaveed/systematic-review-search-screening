"""
full_text_screening_claude.py
=======================

Batched full-text screening of research papers with Claude Opus 4.8.

WORKFLOW (three steps, run as separate commands so you can walk away
while the batch processes -- batches can take up to 24h, though most
finish much faster):

    1. python full_text_screening_claude.py submit
    2. python full_text_screening_claude.py status
    3. python full_text_screening_claude.py fetch


REQUIREMENTS
------------
    pip install anthropic pandas python-dotenv

ENVIRONMENT
-----------
    Create a .env file next to this script (or in the working directory):

        ANTHROPIC_API_KEY=sk-ant-...

"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

try:
    import anthropic
except ImportError:
    sys.exit("Missing dependency. Run: pip install anthropic pandas")

# --------------------------------------------------------------------------
# Configuration -- edit these to match your environment
# --------------------------------------------------------------------------

CSV_PATH = Path("full_text.csv")
PROMPT_PATH = Path("full-text-prompt.txt")

PDF_FOLDERS = [
    Path("paper-pdfs")
]

MODEL = "claude-opus-4-8"
EFFORT = "high"
MAX_TOKENS = 16384


FILE_ID_CACHE_PATH = Path("pdf_file_id_cache.json")
MANIFEST_PATH = Path("screening_manifest.json")
BATCH_IDS_PATH = Path("batch_ids.json")


MAX_REQUESTS_PER_BATCH = 5000

# Safety default for the initial test run. Use --limit N to change it,
# or --all to submit every screenable article.
DEFAULT_TEST_LIMIT = 30

FILES_BETA = "files-api-2025-04-14"


def ROW_FILTER(row: pd.Series) -> bool:
    """Return True if this row should be screened. Edit as needed."""
    pdf = row.get("PDF_Filename")
    return isinstance(pdf, str) and pdf.strip() != ""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_json(path: Path, default):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return default


def save_json(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def find_pdf(filename: str) -> Optional[Path]:
    """Search the configured folders for an exact, then case-insensitive, match."""
    for folder in PDF_FOLDERS:
        candidate = folder / filename
        if candidate.exists():
            return candidate
    # case-insensitive fallback
    target = filename.lower()
    for folder in PDF_FOLDERS:
        if not folder.exists():
            continue
        for f in folder.iterdir():
            if f.name.lower() == target:
                return f
    return None


def get_client() -> anthropic.Anthropic:
    load_dotenv()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        sys.exit(
            "ANTHROPIC_API_KEY was not found. Add it to a .env file, e.g.\n"
            "ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=key)


def load_prompt() -> str:
    if not PROMPT_PATH.exists():
        sys.exit(f"Prompt file not found: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_system_prompt(prompt_template: str, title: str) -> str:
    """Build the stable screening instructions sent through the system field."""
    if "{title}" in prompt_template:
        return prompt_template.format(title=title)
    return prompt_template


def build_user_prompt(title: str) -> str:
    """Keep the per-paper user request short; the PDF is attached beside it."""
    return (
        "Screen the attached full-text paper using the system instructions. "
        f"Paper title: {title or '[title unavailable]'}. "
        "Base the decision on the full methods and results, then return only the required output format."
    )


def parse_decision(response_text: str) -> dict:
    """
    Parse Claude's JSON output robustly.

    Supports either schema:
      {"Include": "Yes", "Reasoning": "...", "Exclusion_Reason": "..."}
    or:
      {"decision": "Include", "reasoning": "...", "exclusion_reason": "..."}
    """
    cleaned = response_text.strip()

    cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\\s*```$", "", cleaned)

    candidates = [cleaned]
    candidates.extend(re.findall(r"\{.*\}", cleaned, flags=re.DOTALL))

    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if not isinstance(data, dict):
            continue

        include_value = data.get("Include", data.get("include", ""))
        decision_value = data.get("decision", data.get("Decision", ""))

        if include_value != "":
            normalized = str(include_value).strip().lower()
            if normalized in {"yes", "include", "included"}:
                decision = "Yes"
            elif normalized in {"no", "exclude", "excluded"}:
                decision = "No"
            else:
                decision = str(include_value)
        else:
            normalized = str(decision_value).strip().lower()
            if normalized in {"include", "included", "yes"}:
                decision = "Yes"
            elif normalized in {"exclude", "excluded", "no"}:
                decision = "No"
            else:
                decision = str(decision_value)

        reasoning = data.get(
            "Reasoning",
            data.get("reasoning", data.get("Rationale", data.get("rationale", ""))),
        )
        exclusion_reason = data.get(
            "Exclusion_Reason",
            data.get("exclusion_reason", data.get("Exclusion Reason", "")),
        )

        if exclusion_reason is None:
            exclusion_reason = ""

        if decision or reasoning or exclusion_reason:
            return {
                "decision": decision,
                "reasoning": reasoning,
                "exclusion_reason": exclusion_reason,
            }

    return {"decision": "", "reasoning": "", "exclusion_reason": ""}


# --------------------------------------------------------------------------
# Step 1: submit
# --------------------------------------------------------------------------

def cmd_submit(args):
    client = get_client()
    prompt_template = load_prompt()

    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    rows = [row for _, row in df.iterrows() if ROW_FILTER(row)]

    if args.offset:
        rows = rows[args.offset:]

    if args.all:
        effective_limit = None
    else:
        effective_limit = args.limit if args.limit is not None else DEFAULT_TEST_LIMIT

    if effective_limit is not None:
        rows = rows[:effective_limit]

    print(f"{len(rows)} rows to screen out of {len(df)} total rows.")

    file_id_cache = load_json(FILE_ID_CACHE_PATH, {})
    manifest = {}
    requests_payload = []
    missing_pdfs = []

    for i, row in enumerate(rows):
        pdf_filename = row["PDF_Filename"].strip()
        pdf_path = find_pdf(pdf_filename)
        if pdf_path is None:
            missing_pdfs.append(pdf_filename)
            continue

        cache_key = str(pdf_path.resolve())
        if cache_key in file_id_cache:
            file_id = file_id_cache[cache_key]
        else:
            print(f"Uploading {pdf_path} ...")
            with open(pdf_path, "rb") as f:
                uploaded = client.beta.files.upload(
                    file=(pdf_path.name, f, "application/pdf")
                )
            file_id = uploaded.id
            file_id_cache[cache_key] = file_id
            save_json(FILE_ID_CACHE_PATH, file_id_cache)  # persist as we go

        custom_id = f"row-{i:06d}"
        manifest[custom_id] = {
            "Normalized_Title": row.get("Normalized_Title", ""),
            "Title": row.get("Title", ""),
            "Database": row.get("Database", ""),
            "PDF_Filename": pdf_filename,
            "PDF_Path": str(pdf_path),
        }

        system_prompt = build_system_prompt(prompt_template, row.get("Title", ""))
        user_prompt = build_user_prompt(row.get("Title", ""))

        requests_payload.append(
            {
                "custom_id": custom_id,
                "params": {
                    "model": MODEL,
                    "max_tokens": MAX_TOKENS,
                    "output_config": {"effort": EFFORT},
                    "system": system_prompt,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "document",
                                    "source": {"type": "file", "file_id": file_id},
                                },
                                {"type": "text", "text": user_prompt},
                            ],
                        }
                    ],
                },
            }
        )

    if missing_pdfs:
        print(f"\nWARNING: {len(missing_pdfs)} PDF(s) not found in configured folders:")
        for m in missing_pdfs[:20]:
            print(f"  - {m}")
        if len(missing_pdfs) > 20:
            print(f"  ... and {len(missing_pdfs) - 20} more")

    if not requests_payload:
        sys.exit("Nothing to submit (no matched PDFs found). Aborting.")

    save_json(MANIFEST_PATH, manifest)

    # Chunk into multiple batches if needed (well under the 100k/256MB API limit)
    batch_ids = load_json(BATCH_IDS_PATH, [])
    for start in range(0, len(requests_payload), MAX_REQUESTS_PER_BATCH):
        chunk = requests_payload[start : start + MAX_REQUESTS_PER_BATCH]
        batch = client.messages.batches.create(
            requests=chunk,
            extra_headers={"anthropic-beta": FILES_BETA},
        )
        print(f"Submitted batch {batch.id} with {len(chunk)} requests.")
        batch_ids.append(batch.id)

    save_json(BATCH_IDS_PATH, batch_ids)
    print(f"\nSaved {len(batch_ids)} batch id(s) to {BATCH_IDS_PATH}.")
    print("Run `python full_text_screening_claude.py status` to check progress.")


# --------------------------------------------------------------------------
# Step 2: status
# --------------------------------------------------------------------------

def cmd_status(args):
    client = get_client()
    batch_ids = load_json(BATCH_IDS_PATH, [])
    if not batch_ids:
        sys.exit("No batch_ids.json found. Run `submit` first.")

    all_ended = True
    for bid in batch_ids:
        b = client.messages.batches.retrieve(bid, extra_headers={"anthropic-beta": FILES_BETA})
        print(f"{bid}: {b.processing_status} -- {b.request_counts}")
        if b.processing_status != "ended":
            all_ended = False

    if all_ended:
        print("\nAll batches finished. Run `python full_text_screening_claude.py fetch`.")
    else:
        print("\nStill processing. Batches can take up to 24h (often much less).")


# --------------------------------------------------------------------------
# Step 3: fetch
# --------------------------------------------------------------------------

def cmd_fetch(args):
    client = get_client()
    batch_ids = load_json(BATCH_IDS_PATH, [])
    manifest = load_json(MANIFEST_PATH, {})
    if not batch_ids:
        sys.exit("No batch_ids.json found. Run `submit` first.")
    if not manifest:
        sys.exit("No screening_manifest.json found. Run `submit` first.")

    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    results_by_custom_id = {}

    for bid in batch_ids:
        b = client.messages.batches.retrieve(bid, extra_headers={"anthropic-beta": FILES_BETA})
        if b.processing_status != "ended":
            sys.exit(f"Batch {bid} is not finished yet ({b.processing_status}). Run `status` first.")

        for result in client.messages.batches.results(bid, extra_headers={"anthropic-beta": FILES_BETA}):
            custom_id = result.custom_id
            if result.result.type == "succeeded":
                message = result.result.message
                text = "".join(
                    block.text for block in message.content if block.type == "text"
                )
                parsed = parse_decision(text)
                results_by_custom_id[custom_id] = {
                    "Screening_Decision": parsed["decision"],
                    "Screening_Reasoning": parsed["reasoning"],
                    "Exclusion_Reason": parsed["exclusion_reason"],
                    "Raw_Response": text,
                    "Screening_Error": "",
                }
            else:
                # errored / canceled / expired
                error_msg = getattr(result.result, "error", result.result.type)
                results_by_custom_id[custom_id] = {
                    "Screening_Decision": "",
                    "Screening_Reasoning": "",
                    "Exclusion_Reason": "",
                    "Raw_Response": "",
                    "Screening_Error": str(error_msg),
                }


    lookup_by_pdf = {}
    for custom_id, meta in manifest.items():
        lookup_by_pdf[meta["PDF_Filename"]] = results_by_custom_id.get(custom_id, {})

    for col in ["Screening_Decision", "Screening_Reasoning", "Exclusion_Reason",
                "Raw_Response", "Screening_Error"]:
        df[col] = df["PDF_Filename"].map(lambda p: lookup_by_pdf.get(p, {}).get(col, ""))

    out_path = Path(args.output)
    df.to_csv(out_path, index=False)
    n_done = sum(1 for v in results_by_custom_id.values() if v.get("Screening_Decision"))
    n_err = sum(1 for v in results_by_custom_id.values() if v.get("Screening_Error"))
    print(f"Wrote {out_path} ({len(df)} rows). {n_done} screened successfully, {n_err} errors.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_submit = sub.add_parser("submit", help="Upload PDFs and submit the batch job.")
    p_submit.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"Submit only the first N screenable rows (default: {DEFAULT_TEST_LIMIT}).",
    )
    p_submit.add_argument(
        "--all",
        action="store_true",
        help="Submit all screenable rows instead of the default 30-paper test run.",
    )
    p_submit.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N screenable rows before submitting.",
    )
    p_submit.set_defaults(func=cmd_submit)

    p_status = sub.add_parser("status", help="Check batch processing status.")
    p_status.set_defaults(func=cmd_status)

    p_fetch = sub.add_parser("fetch", help="Fetch results once processing has ended.")
    p_fetch.add_argument("--output", default="screening_results_claude_remaining_2.csv", help="Output CSV path.")
    p_fetch.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
