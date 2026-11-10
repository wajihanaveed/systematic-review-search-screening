"""
full_text_screening_openai.py
=============================

Batched full-text screening of research papers with OpenAI GPT-5.6 Sol,
high reasoning effort, and high-detail PDF processing.

WORKFLOW
--------
    1. python full_text_screening_openai.py submit
    2. python full_text_screening_openai.py status
    3. python full_text_screening_openai.py fetch

Examples
--------
Test the first 30 screenable papers:
    python full_text_screening_openai.py submit

Submit all screenable papers:
    python full_text_screening_openai.py submit --all

Submit a selected range:
    python full_text_screening_openai.py submit --offset 30 --limit 100

Check progress:
    python full_text_screening_openai.py status

Fetch results:
    python full_text_screening_openai.py fetch --output screening_results_openai.csv

REQUIREMENTS
------------
    pip install -U openai pandas python-dotenv

ENVIRONMENT
-----------
Create a .env file in the working directory:

    OPENAI_API_KEY=sk-...

INPUTS
------
- full_text_already_available.csv
- full-text-prompt.txt
- PDFs located in:
    paper-pdfs/arxiv-papers
    paper-pdfs/OSF-papers
    paper-pdfs/outdated-published-papers
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing dependency. Run: pip install -U openai pandas python-dotenv")


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CSV_PATH = Path("full_text.csv")
PROMPT_PATH = Path("full-text-prompt.txt")

PDF_FOLDERS = [
    Path("paper-pdfs"),
]

MODEL = "gpt-5.6-sol"

REASONING_EFFORT = "high"

PDF_DETAIL = "high"

MAX_OUTPUT_TOKENS = 16384

FILE_ID_CACHE_PATH = Path("openai_pdf_file_id_cache.json")
MANIFEST_PATH = Path("openai_screening_manifest.json")
BATCH_IDS_PATH = Path("openai_batch_ids.json")
BATCH_INPUT_DIR = Path("openai_batch_inputs")
RAW_OUTPUT_DIR = Path("openai_batch_outputs")


MAX_REQUESTS_PER_BATCH = 1000

DEFAULT_TEST_LIMIT = 2


def row_filter(row: pd.Series) -> bool:
    """A row is screenable when PDF_Filename is non-empty."""
    pdf = row.get("PDF_Filename", "")
    # pdf = row.get("PDF_Filename", "")
    return isinstance(pdf, str) and bool(pdf.strip())


# --------------------------------------------------------------------------
# General helpers
# --------------------------------------------------------------------------

def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        sys.exit(f"Could not read {path}: {exc}")


def save_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit(
            "OPENAI_API_KEY was not found.\n"
            "Add this to your .env file:\n"
            "OPENAI_API_KEY=sk-..."
        )
    return OpenAI(api_key=api_key)


def load_prompt() -> str:
    if not PROMPT_PATH.exists():
        sys.exit(f"Prompt file not found: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def find_pdf(filename: str) -> Optional[Path]:
    """Find an exact or case-insensitive filename match."""
    filename = filename.strip()

    for folder in PDF_FOLDERS:
        candidate = folder / filename
        if candidate.is_file():
            return candidate

    target = filename.casefold()
    for folder in PDF_FOLDERS:
        if not folder.is_dir():
            continue
        for path in folder.iterdir():
            if path.is_file() and path.name.casefold() == target:
                return path

    return None


def build_instructions(prompt_template: str, title: str) -> str:
    """
    Insert {title} only when the screening prompt contains that placeholder.

    Using replace rather than str.format prevents other JSON braces in the
    prompt from being interpreted as Python formatting placeholders.
    """
    return prompt_template.replace("{title}", title)


def build_user_text(title: str) -> str:
    return (
        "Screen the attached full-text paper using the supplied screening "
        "instructions. Base the decision on the complete methods and results, "
        "not only the abstract, introduction, or discussion. "
        f"Paper title: {title or '[title unavailable]'}. "
        "Return only the required JSON output."
    )


# --------------------------------------------------------------------------
# OpenAI file and response helpers
# --------------------------------------------------------------------------

def upload_pdf_once(
    client: OpenAI,
    pdf_path: Path,
    file_id_cache: dict[str, str],
) -> str:
    """
    Upload a PDF once with purpose=user_data and cache its OpenAI file ID.
    """
    cache_key = str(pdf_path.resolve())

    cached_id = file_id_cache.get(cache_key)
    if cached_id:
        return cached_id

    print(f"Uploading {pdf_path} ...")
    with pdf_path.open("rb") as file_handle:
        uploaded = client.files.create(
            file=file_handle,
            purpose="user_data",
        )

    file_id_cache[cache_key] = uploaded.id
    save_json(FILE_ID_CACHE_PATH, file_id_cache)
    return uploaded.id


def extract_response_text(response_body: dict[str, Any]) -> str:
    """
    Extract assistant output text from a Responses API response body.

    Expected structure:
      body["output"] -> message items -> content items of type "output_text"
    """
    texts: list[str] = []

    for output_item in response_body.get("output", []) or []:
        if not isinstance(output_item, dict):
            continue

        if output_item.get("type") != "message":
            continue

        for content_item in output_item.get("content", []) or []:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") == "output_text":
                text = content_item.get("text", "")
                if isinstance(text, str):
                    texts.append(text)


    if not texts:
        possible_text = response_body.get("output_text")
        if isinstance(possible_text, str):
            texts.append(possible_text)

    return "".join(texts).strip()


def parse_decision(response_text: str) -> dict:
    """
    Parse common screening JSON schemas.

    Supported examples:
      {"Include": "Yes", "Reasoning": "...", "Exclusion_Reason": "..."}
      {"decision": "Include", "reasoning": "...", "exclusion_reason": "..."}
      {"decision": "Yes", "reasoning": "..."}
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



def parse_jsonl_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            print(
                f"WARNING: Could not parse output JSONL line "
                f"{line_number}: {exc}"
            )
            continue

        if isinstance(item, dict):
            rows.append(item)

    return rows


# --------------------------------------------------------------------------
# Step 1: submit
# --------------------------------------------------------------------------

def cmd_submit(args: argparse.Namespace) -> None:
    client = get_client()
    prompt_template = load_prompt()

    if not CSV_PATH.exists():
        sys.exit(f"CSV file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")

    selected: list[tuple[int, pd.Series]] = [
        (int(index), row)
        for index, row in df.iterrows()
        if row_filter(row)
    ]

    if args.offset:
        selected = selected[args.offset :]

    if args.all:
        effective_limit = None
    else:
        effective_limit = (
            args.limit if args.limit is not None else DEFAULT_TEST_LIMIT
        )

    if effective_limit is not None:
        selected = selected[:effective_limit]

    print(
        f"{len(selected)} rows selected for screening "
        f"out of {len(df)} total CSV rows."
    )

    if not selected:
        sys.exit("No screenable rows were selected.")

    BATCH_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    file_id_cache: dict[str, str] = load_json(FILE_ID_CACHE_PATH, {})
    manifest: dict[str, dict[str, Any]] = {}
    batch_requests: list[dict[str, Any]] = []
    missing_pdfs: list[str] = []

    for original_index, row in selected:
        pdf_filename = row.get("PDF_Filename", "").strip()
        pdf_path = find_pdf(pdf_filename)

        if pdf_path is None:
            missing_pdfs.append(pdf_filename)
            continue

        file_id = upload_pdf_once(client, pdf_path, file_id_cache)
        custom_id = f"row-{original_index:08d}"

        manifest[custom_id] = {
            "Original_Row_Index": original_index,
            "Normalized_Title": row.get("Normalized_Title", ""),
            "Title": row.get("Title", ""),
            "Database": row.get("Database", ""),
            "PDF_Filename": pdf_filename,
            "PDF_Path": str(pdf_path),
            "OpenAI_File_ID": file_id,
        }

        instructions = build_instructions(
            prompt_template,
            row.get("Title", ""),
        )
        user_text = build_user_text(row.get("Title", ""))

        batch_requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": MODEL,
                    "instructions": instructions,
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_file",
                                    "file_id": file_id,
                                    "detail": PDF_DETAIL,
                                },
                                {
                                    "type": "input_text",
                                    "text": user_text,
                                },
                            ],
                        }
                    ],
                    "reasoning": {
                        "effort": REASONING_EFFORT,
                    },
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },
            }
        )

    if missing_pdfs:
        print(
            f"\nWARNING: {len(missing_pdfs)} PDF(s) were not found "
            "in the configured folders:"
        )
        for filename in missing_pdfs[:20]:
            print(f"  - {filename}")
        if len(missing_pdfs) > 20:
            print(f"  ... and {len(missing_pdfs) - 20} more")

    if not batch_requests:
        sys.exit("Nothing to submit because no matching PDFs were found.")

    save_json(MANIFEST_PATH, manifest)
    submitted_batches: list[dict[str, Any]] = []

    for chunk_number, start in enumerate(
        range(0, len(batch_requests), MAX_REQUESTS_PER_BATCH),
        start=1,
    ):
        chunk = batch_requests[start : start + MAX_REQUESTS_PER_BATCH]
        input_path = BATCH_INPUT_DIR / (
            f"openai_batch_input_{chunk_number:03d}.jsonl"
        )

        with input_path.open("w", encoding="utf-8") as output_file:
            for request in chunk:
                output_file.write(
                    json.dumps(request, ensure_ascii=False) + "\n"
                )

        input_size_mb = input_path.stat().st_size / (1024 * 1024)
        print(
            f"Prepared {input_path} with {len(chunk)} requests "
            f"({input_size_mb:.2f} MB)."
        )

        if input_size_mb >= 200:
            sys.exit(
                f"{input_path} is {input_size_mb:.2f} MB, which reaches "
                "the Batch API upload limit. Reduce "
                "MAX_REQUESTS_PER_BATCH and submit again."
            )

        with input_path.open("rb") as batch_file:
            uploaded_batch_input = client.files.create(
                file=batch_file,
                purpose="batch",
            )

        batch = client.batches.create(
            input_file_id=uploaded_batch_input.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={
                "description": (
                    "Full-text systematic-review screening with "
                    "GPT-5.6 Sol, high reasoning"
                )
            },
        )

        submitted_batches.append(
            {
                "batch_id": batch.id,
                "input_file_id": uploaded_batch_input.id,
                "local_input_path": str(input_path),
                "request_count": len(chunk),
            }
        )
        print(
            f"Submitted batch {batch.id} with {len(chunk)} requests."
        )

    save_json(BATCH_IDS_PATH, submitted_batches)

    print(
        f"\nSaved {len(submitted_batches)} batch record(s) "
        f"to {BATCH_IDS_PATH}."
    )
    print(
        "Run `python full_text_screening_openai.py status` "
        "to check progress."
    )


# --------------------------------------------------------------------------
# Step 2: status
# --------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    client = get_client()
    batch_records = load_json(BATCH_IDS_PATH, [])

    if not batch_records:
        sys.exit(
            f"No batches found in {BATCH_IDS_PATH}. Run `submit` first."
        )

    all_completed = True

    for record in batch_records:
        batch_id = (
            record["batch_id"] if isinstance(record, dict) else record
        )
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts

        print(
            f"{batch.id}: {batch.status} -- "
            f"total={getattr(counts, 'total', 0)}, "
            f"completed={getattr(counts, 'completed', 0)}, "
            f"failed={getattr(counts, 'failed', 0)}"
        )

        if batch.status != "completed":
            all_completed = False

        if batch.errors:
            print(f"  Validation/API errors: {batch.errors}")

    if all_completed:
        print(
            "\nAll batches completed. Run "
            "`python full_text_screening_openai.py fetch`."
        )
    else:
        print(
            "\nAt least one batch is not completed yet. "
            "Run this status command again later."
        )


# --------------------------------------------------------------------------
# Step 3: fetch
# --------------------------------------------------------------------------

def read_batch_output(
    client: OpenAI,
    batch_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    batch = client.batches.retrieve(batch_id)

    if batch.status != "completed":
        sys.exit(
            f"Batch {batch_id} is not completed; current status is "
            f"{batch.status}. Run `status` first."
        )

    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    successful_lines: list[dict[str, Any]] = []
    error_lines: list[dict[str, Any]] = []

    if batch.output_file_id:
        output_response = client.files.content(batch.output_file_id)
        output_text = output_response.text

        output_path = RAW_OUTPUT_DIR / f"{batch_id}_output.jsonl"
        output_path.write_text(output_text, encoding="utf-8")
        successful_lines = parse_jsonl_text(output_text)

    if batch.error_file_id:
        error_response = client.files.content(batch.error_file_id)
        error_text = error_response.text

        error_path = RAW_OUTPUT_DIR / f"{batch_id}_errors.jsonl"
        error_path.write_text(error_text, encoding="utf-8")
        error_lines = parse_jsonl_text(error_text)

    return successful_lines, error_lines


def cmd_fetch(args: argparse.Namespace) -> None:
    client = get_client()
    batch_records = load_json(BATCH_IDS_PATH, [])
    manifest = load_json(MANIFEST_PATH, {})

    if not batch_records:
        sys.exit(
            f"No batches found in {BATCH_IDS_PATH}. Run `submit` first."
        )
    if not manifest:
        sys.exit(
            f"No manifest found in {MANIFEST_PATH}. Run `submit` first."
        )

    if not CSV_PATH.exists():
        sys.exit(f"CSV file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, dtype=str).fillna("")
    results_by_custom_id: dict[str, dict[str, str]] = {}

    for record in batch_records:
        batch_id = (
            record["batch_id"] if isinstance(record, dict) else record
        )
        success_lines, error_lines = read_batch_output(client, batch_id)

        for item in success_lines:
            custom_id = str(item.get("custom_id", ""))
            response = item.get("response") or {}
            status_code = response.get("status_code")
            body = response.get("body") or {}
            top_level_error = item.get("error")

            if status_code == 200 and not top_level_error:
                raw_text = extract_response_text(body)
                parsed = parse_decision(raw_text)

                results_by_custom_id[custom_id] = {
                    "OpenAI_Screening_Decision": parsed["decision"],
                    "OpenAI_Screening_Reasoning": parsed["reasoning"],
                    "OpenAI_Exclusion_Reason": parsed["exclusion_reason"],
                    "OpenAI_Raw_Response": raw_text,
                    "OpenAI_Screening_Error": "",
                    "OpenAI_Response_ID": str(body.get("id", "")),
                }
            else:
                error_text = json.dumps(
                    top_level_error or body or response,
                    ensure_ascii=False,
                )
                results_by_custom_id[custom_id] = {
                    "OpenAI_Screening_Decision": "",
                    "OpenAI_Screening_Reasoning": "",
                    "OpenAI_Exclusion_Reason": "",
                    "OpenAI_Raw_Response": "",
                    "OpenAI_Screening_Error": error_text,
                    "OpenAI_Response_ID": "",
                }

        for item in error_lines:
            custom_id = str(item.get("custom_id", ""))
            error_value = item.get("error") or item
            results_by_custom_id[custom_id] = {
                "OpenAI_Screening_Decision": "",
                "OpenAI_Screening_Reasoning": "",
                "OpenAI_Exclusion_Reason": "",
                "OpenAI_Raw_Response": "",
                "OpenAI_Screening_Error": json.dumps(
                    error_value,
                    ensure_ascii=False,
                ),
                "OpenAI_Response_ID": "",
            }

    result_columns = [
        "OpenAI_Screening_Decision",
        "OpenAI_Screening_Reasoning",
        "OpenAI_Exclusion_Reason",
        "OpenAI_Raw_Response",
        "OpenAI_Screening_Error",
        "OpenAI_Response_ID",
    ]

    for column in result_columns:
        if column not in df.columns:
            df[column] = ""

    for custom_id, metadata in manifest.items():
        row_index = metadata.get("Original_Row_Index")
        result = results_by_custom_id.get(custom_id)

        if result is None:
            result = {
                "OpenAI_Screening_Decision": "",
                "OpenAI_Screening_Reasoning": "",
                "OpenAI_Exclusion_Reason": "",
                "OpenAI_Raw_Response": "",
                "OpenAI_Screening_Error": (
                    "No output line was returned for this request."
                ),
                "OpenAI_Response_ID": "",
            }

        if not isinstance(row_index, int) or not (0 <= row_index < len(df)):
            print(
                f"WARNING: Invalid original row index for {custom_id}: "
                f"{row_index}"
            )
            continue

        for column in result_columns:
            df.at[row_index, column] = result.get(column, "")

    output_path = Path(args.output)
    df.to_csv(output_path, index=False)

    successful = sum(
        1
        for result in results_by_custom_id.values()
        if result.get("OpenAI_Screening_Decision")
    )
    errors = sum(
        1
        for result in results_by_custom_id.values()
        if result.get("OpenAI_Screening_Error")
    )
    unparsed = sum(
        1
        for result in results_by_custom_id.values()
        if result.get("OpenAI_Raw_Response")
        and not result.get("OpenAI_Screening_Decision")
        and not result.get("OpenAI_Screening_Error")
    )

    print(
        f"Wrote {output_path} ({len(df)} rows). "
        f"{successful} decisions parsed successfully, "
        f"{errors} request errors, "
        f"{unparsed} responses returned but could not be parsed."
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    submit_parser = subparsers.add_parser(
        "submit",
        help="Upload PDFs and submit OpenAI batch jobs.",
    )
    submit_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Submit only the first N screenable rows after the offset "
            f"(default test limit: {DEFAULT_TEST_LIMIT})."
        ),
    )
    submit_parser.add_argument(
        "--all",
        action="store_true",
        help="Submit all screenable rows.",
    )
    submit_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N screenable rows.",
    )
    submit_parser.set_defaults(func=cmd_submit)

    status_parser = subparsers.add_parser(
        "status",
        help="Check OpenAI batch progress.",
    )
    status_parser.set_defaults(func=cmd_status)

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Download and merge completed OpenAI batch results.",
    )
    fetch_parser.add_argument(
        "--output",
        default="screening_results_openai_remaining_2.csv",
        help="Output CSV path.",
    )
    fetch_parser.set_defaults(func=cmd_fetch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
