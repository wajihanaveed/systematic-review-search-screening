import pandas as pd
import json
import time
import requests
import os
from dotenv import load_dotenv

# ============================================================
# CONFIGURATION
# ============================================================
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
INPUT_FILE = "add your input file here"  # e.g., "wos_export.csv"
OUTPUT_FILE = "screening_results.csv"
BATCH_INPUT_FILE = "screening_batch_input.jsonl"
MODEL = "gpt-5.6-sol"

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


def get_abstract(row):
    abstract = str(row.get("Abstract Note", "")).strip()
    if not abstract or abstract.lower() == "nan":
        abstract = str(row.get("Abstract-Retrieved", "")).strip()
    if not abstract or abstract.lower() == "nan":
        return ""
    return abstract


def get_title(row):
    title = str(row.get("Title", "")).strip()
    if not title or title.lower() == "nan":
        return ""
    return title


# ============================================================
# STEP 1: CREATE BATCH INPUT FILE
# ============================================================
def create_batch_file(df):
    print(f"Creating batch input file for {len(df)} rows...")

    with open(BATCH_INPUT_FILE, "w") as f:
        for idx, row in df.iterrows():
            title = get_title(row)
            abstract = get_abstract(row)
            item_type = str(row.get("Item Type", "")).strip()

            user_message = f"""PAPER TO SCREEN

Title: {title if title else '[No title available]'}

Abstract: {abstract if abstract else '[No abstract available]'}"""

            request = {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ],

                }
            }
            f.write(json.dumps(request) + "\n")

    print(f"Batch file created: {BATCH_INPUT_FILE}")


# ============================================================
# STEP 2: UPLOAD FILE AND SUBMIT BATCH
# ============================================================
def submit_batch():
    headers = {"Authorization": f"Bearer {API_KEY}"}

    print("Uploading batch file to OpenAI...")
    with open(BATCH_INPUT_FILE, "rb") as f:
        response = requests.post(
            "https://api.openai.com/v1/files",
            headers=headers,
            files={"file": (BATCH_INPUT_FILE, f, "application/jsonl")},
            data={"purpose": "batch"}
        )

    if response.status_code != 200:
        print(f"Upload failed: {response.text}")
        return None

    file_id = response.json()["id"]
    print(f"File uploaded. ID: {file_id}")

    print("Submitting batch job...")
    response = requests.post(
        "https://api.openai.com/v1/batches",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h"
        }
    )

    if response.status_code != 200:
        print(f"Batch submission failed: {response.text}")
        return None

    batch_id = response.json()["id"]
    print(f"Batch submitted. ID: {batch_id}")
    print(f"Save this ID: {batch_id}")

    with open("batch_id.txt", "w") as f:
        f.write(batch_id)

    return batch_id


# ============================================================
# STEP 3: POLL STATUS
# ============================================================
def check_status(batch_id):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.get(
        f"https://api.openai.com/v1/batches/{batch_id}",
        headers=headers
    )
    return response.json()


def wait_for_completion(batch_id):
    print(f"Polling batch status for {batch_id}...")

    while True:
        status_data = check_status(batch_id)
        status = status_data.get("status")
        completed = status_data.get("request_counts", {}).get("completed", 0)
        total = status_data.get("request_counts", {}).get("total", 0)

        print(f"  Status: {status} | Completed: {completed}/{total}")

        if status == "completed":
            print("Batch completed.")
            return status_data
        elif status in ["failed", "expired", "cancelled"]:
            print(f"Batch ended with status: {status}")
            print(json.dumps(status_data, indent=2))
            return None

        time.sleep(120)


# ============================================================
# STEP 4: RETRIEVE AND MERGE RESULTS
# ============================================================
def retrieve_results(batch_id, df):
    headers = {"Authorization": f"Bearer {API_KEY}"}

    status_data = check_status(batch_id)
    output_file_id = status_data.get("output_file_id")

    if not output_file_id:
        print("No output file found. Batch may not be complete yet.")
        return

    print(f"Downloading results. Output file ID: {output_file_id}")
    response = requests.get(
        f"https://api.openai.com/v1/files/{output_file_id}/content",
        headers=headers
    )

    for col in ["Include", "Exclusion_Reason", "AI_Type",
                "Outcome_Relevant", "Reasoning"]:
        if col not in df.columns:
            df[col] = None

    success = 0
    errors = 0

    for line in response.text.strip().split("\n"):
        if not line:
            continue
        try:
            result = json.loads(line)
            idx = int(result["custom_id"])
            content = result["response"]["body"]["choices"][0]["message"]["content"]

            # Strip markdown fences if model returns them despite instruction
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()

            parsed = json.loads(content)

            df.at[idx, "Include"] = parsed.get("Include", "ERROR")
            df.at[idx, "Exclusion_Reason"] = parsed.get("Exclusion_Reason")
            df.at[idx, "AI_Type"] = parsed.get("AI_Type")
            df.at[idx, "Outcome_Relevant"] = parsed.get("Outcome_Relevant")
            df.at[idx, "Reasoning"] = parsed.get("Reasoning")
            success += 1

        except Exception as e:
            errors += 1
            print(f"  Error on custom_id {result.get('custom_id')}: {e}")

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
# STEP 5: CHECK ERROR FILE
# ============================================================
def retrieve_errors(batch_id):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    status_data = check_status(batch_id)
    error_file_id = status_data.get("error_file_id")

    if not error_file_id:
        print("No error file found yet.")
        return

    print(f"Downloading error file: {error_file_id}")
    response = requests.get(
        f"https://api.openai.com/v1/files/{error_file_id}/content",
        headers=headers
    )

    lines = response.text.strip().split("\n")
    print(f"Total error lines: {len(lines)}")
    print("First 5 errors:")
    for line in lines[:5]:
        try:
            print(json.dumps(json.loads(line), indent=2))
        except:
            print(line)


# ============================================================
# MAIN
# ============================================================

def main():
    import sys

    df = pd.read_csv(INPUT_FILE, dtype=str)
    print(f"Loaded {len(df)} rows from {INPUT_FILE}")

    if len(sys.argv) < 2:
        print("""
            Usage:
            python screen_batch_gpt.py submit              — create and submit batch
            python screen_batch_gpt.py status <batch_id>  — check batch status
            python screen_batch_gpt.py wait <batch_id>    — poll until done then retrieve
            python screen_batch_gpt.py retrieve <batch_id>— download and merge results
            python screen_batch_gpt.py errors <batch_id>  — inspect error file
            python screen_batch_gpt.py cancel <batch_id>  — cancel batch
            python screen_batch_gpt.py test               — test single request
        """)
        return

    command = sys.argv[1]

    if command == "submit":
        create_batch_file(df)
        submit_batch()

    elif command == "status":
        if len(sys.argv) < 3:
            print("Usage: python screen_batch_gpt.py status <batch_id>")
            return
        batch_id = sys.argv[2]
        print(json.dumps(check_status(batch_id), indent=2))

    elif command == "retrieve":
        if len(sys.argv) < 3:
            print("Usage: python screen_batch_gpt.py retrieve <batch_id>")
            return
        batch_id = sys.argv[2]
        retrieve_results(batch_id, df)

    elif command == "wait":
        if len(sys.argv) < 3:
            print("Usage: python screen_batch_gpt.py wait <batch_id>")
            return
        batch_id = sys.argv[2]
        result = wait_for_completion(batch_id)
        if result:
            retrieve_results(batch_id, df)

    elif command == "errors":
        if len(sys.argv) < 3:
            print("Usage: python screen_batch_gpt.py errors <batch_id>")
            return
        batch_id = sys.argv[2]
        retrieve_errors(batch_id)

    elif command == "cancel":
        if len(sys.argv) < 3:
            print("Usage: python screen_batch_gpt.py cancel <batch_id>")
            return
        batch_id = sys.argv[2]
        headers = {"Authorization": f"Bearer {API_KEY}"}
        response = requests.post(
            f"https://api.openai.com/v1/batches/{batch_id}/cancel",
            headers=headers
        )
        print(json.dumps(response.json(), indent=2))

    elif command == "test":
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "PAPER TO SCREEN\n\nTitle: Test paper about ChatGPT and loneliness\n\nAbstract: This study examines whether ChatGPT reduces loneliness in older adults."}
            ]
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()