import pandas as pd
import json
import time
import os
from dotenv import load_dotenv
import anthropic

# ============================================================
# CONFIGURATION
# ============================================================
load_dotenv()
client = anthropic.Anthropic()

INPUT_FILE = "add your input file here"  # e.g., "wos_export.ris"
OUTPUT_FILE = "add your output file here"  # e.g., "wos_screened.csv"
BATCH_INPUT_FILE = "add your batch input file here"  # e.g., "wos_batch.jsonl"
MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """You are a systematic review screener. Your task is to screen a paper
for inclusion or exclusion based on title and abstract only.

INCLUSION CRITERIA
Retain the paper (Include = Yes) if ALL of the following are met:

1. Reports original empirical findings, is not a review, meta-analysis,
   commentary or theoretical paper

2. Involves human participants interacting with an AI system, including:
   - Chatbots
   - Conversational agents
   - Generative AI tools
   - Large language models (LLMs)
   - AI companions
   - Voice assistants
   - Or closely related conversational/generative technologies
   
   EXPLICITLY EXCLUDED AI types:
   - Rule-based chatbots
   - Decision-tree chatbots
   - Scripted conversational agents
   - Retrieval-based chatbots with no generative component
   
   If the AI type is ambiguous and cannot be determined from the title 
   or abstract, mark as UNCERTAIN

3. Studies at least one social, interpersonal or psychosocial outcome,
   including but not limited to:
   - Loneliness
   - Social isolation
   - Social connectedness
   - Belonging
   - Social support
   - Social satisfaction
   - Interpersonal well-being
   - Relationship formation
   - Companionship
   - Anthropomorphism
   - Emotional attachment
   - And related constructs

   OR studies predictors of AI use where those predictors are social
   in nature, such as:
   - Loneliness
   - Social isolation
   - Unmet belonging needs
   - Social support deficits

4. Medical and clinical generative AI chatbot studies ARE included if
   they involve direct human use of a generative AI system and report
   relevant social, emotional, psychological or well-being outcomes,
   for example:
   - Conversational agents designed to reduce loneliness
   - AI companions providing companionship
   - Chatbots enhancing social support
   - Interventions improving psychosocial well-being

EXCLUSION CRITERIA
Exclude the paper (Include = No) if ANY of the following apply:

(a) Does not involve human participants

(b) Focuses solely on technical development, algorithmic performance,
    model architecture or system design without examining user outcomes

(c) Discusses AI conceptually without reporting empirical data

(d) Focuses exclusively on educational, productivity or
    task-performance outcomes with no social or interpersonal component

(e) Examines robotics, embodied AI or other non-conversational
    technologies. 

(f) Is a thesis or dissertation

(g) Is a book, book chapter or edited volume

(h) Is an entire conference proceedings volume as opposed to an
    individual conference paper with its own abstract

(i) Abstract is missing AND the publication type suggests an abstract
    should be present (e.g. journal article, individual conference
    paper), do NOT exclude book chapters or proceedings volumes
    under this code since missing abstracts are expected for those
    types; exclude those under (g) or (h) instead

EXCLUSION REASON CODES
Use exactly these codes in your Exclusion_Reason field:
(a) No human participants
(b) Technical/system paper
(c) Rule-based or non-generative AI
(d) Conceptual/theoretical, no empirical data
(e) Educational/productivity outcomes only, no social component
(f) Robotics/embodied/non-conversational AI
(g) Thesis or dissertation
(h) Book or book chapter
(i) Entire proceedings volume, no individual abstract
(j) Missing abstract, expected to be present for this publication type

DECISION RULES
- Base your decision on title and abstract only, do not infer or
  assume outcomes not explicitly mentioned
- If the abstract is missing:
    - If publication type is a journal article or individual paper:
      Include = No, Exclusion_Reason = (j) Missing abstract
    - If publication type is a book, book chapter, or entire 
      proceedings volume: exclude under (g) or (h) respectively
- If the paper might meet criteria but you are not certain, output:
  Include = UNCERTAIN and explain why in your reasoning
- If the AI type is ambiguous (e.g. "digital intervention", "chatbot",
  "conversational agent" with no further specification of whether it
  is generative or rule-based), mark as UNCERTAIN
- Mental health chatbot studies are included ONLY if the chatbot is
  generative AI based and reports depression, loneliness or
  psychosocial wellbeing outcomes
- Studies reporting ONLY clinical outcomes (e.g. medication adherence,
  blood pressure) with no psychosocial component are EXCLUDED under (d)
- Studies about attitudes toward or perceptions of AI without direct
  human-AI interaction are EXCLUDED under (c)

OUTPUT FORMAT
Return your response as a JSON object with exactly these fields:

{
  "Include": "Yes / No / UNCERTAIN",
  "Exclusion_Reason": "code and label if No, null if Yes or UNCERTAIN",
  "AI_Type": "specific AI system mentioned and whether it is 
              generative or rule-based, or UNCERTAIN if unclear",
  "Outcome_Relevant": "specific social/psychosocial outcome(s)
                        mentioned, or null if none identified",
  "Reasoning": "2-3 sentences explaining your decision, referencing
                the specific inclusion or exclusion criterion met"
}"""


# ============================================================
# PARSE RIS FILE
# ============================================================
def parse_ris(filepath):
    records = []
    current = {}

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            # End of record
            if line.strip() == "ER  -":
                if current:
                    records.append(current)
                    current = {}
                continue

            # RIS tag format: "XX  - value"
            if len(line) > 5 and line[2:6] == "  - ":
                tag = line[0:2].strip()
                value = line[6:].strip()

                if tag == "TI":
                    current["Title"] = value
                elif tag == "AB":
                    current["Abstract"] = value
                elif tag == "PY":
                    current["Year"] = value
                elif tag == "AU":
                    current.setdefault("Authors", []).append(value)
                elif tag == "T2":
                    current["Journal"] = value
                elif tag == "DO":
                    current["DOI"] = value
                elif tag == "ID":
                    current["ID"] = value

            # Handle multi-line abstract (continuation lines)
            elif line.startswith("   ") and "Abstract" in current:
                current["Abstract"] += " " + line.strip()

    # Catch last record if file doesn't end with ER
    if current:
        records.append(current)

    # Flatten authors list
    for r in records:
        if "Authors" in r:
            r["Authors"] = "; ".join(r["Authors"])

    return pd.DataFrame(records)


def get_abstract(row):
    abstract = str(row.get("Abstract", "")).strip()
    if not abstract or abstract.lower() == "nan":
        return ""
    return abstract


def get_title(row):
    title = str(row.get("Title", "")).strip()
    if not title or title.lower() == "nan":
        return ""
    return title


# ============================================================
# STEP 1: BUILD BATCH REQUESTS
# ============================================================
def create_batch_file(df):
    """Builds the list of Message Batch requests and also writes them
    to a local .jsonl file for inspection/record-keeping."""
    print(f"Creating batch requests for {len(df)} rows...")

    requests_list = []

    with open(BATCH_INPUT_FILE, "w") as f:
        for idx, row in df.iterrows():
            title = get_title(row)
            abstract = get_abstract(row)

            user_message = f"""PAPER TO SCREEN

Title: {title if title else '[No title available]'}

Abstract: {abstract if abstract else '[No abstract available]'}"""

            request = {
                "custom_id": str(idx),
                "params": {
                    "model": MODEL,
                    "max_tokens": 8192,
                    "system": SYSTEM_PROMPT,
                    "messages": [
                        {"role": "user", "content": user_message}
                    ],
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "low"},
                }
            }
            requests_list.append(request)
            f.write(json.dumps(request) + "\n")

    print(f"Batch requests built and saved to: {BATCH_INPUT_FILE}")
    return requests_list


# ============================================================
# STEP 2: SUBMIT BATCH
# ============================================================
def submit_batch(requests_list):
    print(f"Submitting batch of {len(requests_list)} requests to Anthropic...")
    try:
        message_batch = client.messages.batches.create(requests=requests_list)
    except anthropic.APIError as e:
        print(f"Batch submission failed: {e}")
        return None

    print("\nBatch submitted successfully.")
    print(f"Batch ID: {message_batch.id}")
    print("Copy this ID for all future commands.")
    return message_batch.id


# ============================================================
# STEP 3: STATUS AND POLLING
# ============================================================
def check_status(batch_id):
    return client.messages.batches.retrieve(batch_id)


def wait_for_completion(batch_id):
    print(f"Polling batch {batch_id}...")
    while True:
        batch = check_status(batch_id)
        counts = batch.request_counts
        print(
            f"  Status: {batch.processing_status} | "
            f"Succeeded: {counts.succeeded} | Errored: {counts.errored} | "
            f"Processing: {counts.processing} | "
            f"Canceled: {counts.canceled} | Expired: {counts.expired}"
        )

        if batch.processing_status == "ended":
            print("Batch completed.")
            return batch

        time.sleep(120)


# ============================================================
# STEP 4: RETRIEVE RESULTS
# ============================================================
def _extract_text(message):
    """Pulls the text block out of a Claude message, skipping any
    thinking blocks."""
    for block in message.content:
        if block.type == "text":
            return block.text
    return ""


def retrieve_results(batch_id, df):
    batch = check_status(batch_id)

    if batch.processing_status != "ended":
        print(f"Batch is not finished yet (status: {batch.processing_status}).")
        return

    print(f"Downloading results for batch: {batch_id}")

    for col in ["Include", "Exclusion_Reason", "AI_Type",
                "Outcome_Relevant", "Reasoning"]:
        if col not in df.columns:
            df[col] = None

    success = 0
    errors = 0

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        try:
            idx = int(custom_id)

            if result.result.type != "succeeded":
                raise ValueError(f"Request {result.result.type}")

            content = _extract_text(result.result.message).strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            if not content:
                block_types = [b.type for b in result.result.message.content]
                raise ValueError(
                    f"empty text block (stop_reason="
                    f"{result.result.message.stop_reason}, "
                    f"content blocks={block_types}) — likely truncated "
                    f"by max_tokens, consider raising it"
                )

            parsed = json.loads(content)
            df.at[idx, "Include"] = parsed.get("Include", "ERROR")
            df.at[idx, "Exclusion_Reason"] = parsed.get("Exclusion_Reason")
            df.at[idx, "AI_Type"] = parsed.get("AI_Type")
            df.at[idx, "Outcome_Relevant"] = parsed.get("Outcome_Relevant")
            df.at[idx, "Reasoning"] = parsed.get("Reasoning")
            success += 1

        except Exception as e:
            errors += 1
            print(f"  Error on custom_id {custom_id}: {e}")

    df.to_csv(OUTPUT_FILE, index=False)
    counts = df["Include"].value_counts()
    print(f"""
============================================================
DONE
============================================================
Successfully parsed: {success}
Errors:              {errors}

DECISIONS:
{counts.to_string()}

Output saved to: {OUTPUT_FILE}
============================================================""")


# ============================================================
# STEP 5: ERROR INSPECTION
# ============================================================
def retrieve_errors(batch_id):
    batch = check_status(batch_id)

    if batch.processing_status != "ended":
        print(f"Batch is not finished yet (status: {batch.processing_status}).")
        return

    errored = []
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "errored":
            errored.append(result)

    print(f"Total errored requests: {errored.__len__()}")
    print("First 5 errors:")
    for result in errored[:5]:
        print(f"custom_id: {result.custom_id}")
        print(result.result.error.model_dump_json(indent=2))


# ============================================================
# MAIN
# ============================================================
def main():
    import sys

    print(f"Parsing {INPUT_FILE}...")
    df = parse_ris(INPUT_FILE)
    print(f"Parsed {len(df)} records from RIS file")

    if len(sys.argv) < 2:
        print("""
Usage:
  python screen_batch_[DB_name]_claude.py submit               — parse RIS and submit batch
  python screen_batch_[DB_name]_claude.py status <batch_id>   — check batch status
  python screen_batch_[DB_name]_claude.py wait <batch_id>     — poll until done then retrieve
  python screen_batch_[DB_name]_claude.py retrieve <batch_id> — download and merge results
  python screen_batch_[DB_name]_claude.py errors <batch_id>   — inspect errored requests
  python screen_batch_[DB_name]_claude.py cancel <batch_id>   — cancel batch
  python screen_batch_[DB_name]_claude.py preview             — preview first 3 parsed records
        """)
        return

    command = sys.argv[1]

    if command == "submit":
        requests_list = create_batch_file(df)
        submit_batch(requests_list)

    elif command == "status":
        if len(sys.argv) < 3:
            print("Usage: python screen_batch_[DB_name]_claude.py status <batch_id>")
            return
        batch = check_status(sys.argv[2])
        print(batch.model_dump_json(indent=2))

    elif command == "retrieve":
        if len(sys.argv) < 3:
            print("Usage: python screen_batch_[DB_name]_claude.py retrieve <batch_id>")
            return
        retrieve_results(sys.argv[2], df)

    elif command == "wait":
        if len(sys.argv) < 3:
            print("Usage: python screen_batch_[DB_name]_claude.py wait <batch_id>")
            return
        result = wait_for_completion(sys.argv[2])
        if result:
            retrieve_results(sys.argv[2], df)

    elif command == "errors":
        if len(sys.argv) < 3:
            print("Usage: python screen_batch_[DB_name]_claude.py errors <batch_id>")
            return
        retrieve_errors(sys.argv[2])

    elif command == "cancel":
        if len(sys.argv) < 3:
            print("Usage: python screen_batch_[DB_name]_claude.py cancel <batch_id>")
            return
        batch = client.messages.batches.cancel(sys.argv[2])
        print(batch.model_dump_json(indent=2))

    elif command == "preview":
        cols = [c for c in ["Title", "Abstract",
                             "Year", "Authors"] if c in df.columns]
        print(df[cols].head(3).to_string())


if __name__ == "__main__":
    main()