# Python Scripts for Search and LLM-Assisted Screening for Literature Review

This repository contains the scripts which can be used for literature identification and screening for a systematic review. 

## Repository contents

| File | Description |
| --- | --- |
| `OSF.ipynb` | Notebook for searching and retrieving records from OSF. |
| `arxiv.ipynb` | Notebook for arXiv literature searches. |
| `title_&_abstract_screening_gpt.py` | Batch screening script using GPT. |
| `title_&_abstract_screening_claude.py` | Batch screening script using Claude. |
| `full_text_screening_gpt.py` | Full-text screening script using OpenAI models. |
| `full_text_screening_claude.py` | Full-text screening script using Claude. |

## Using the code

1. Clone this repository and create a Python virtual environment.
2. Install the Python packages imported by the notebooks and scripts you intend to run.
3. Configure any required model API credentials locally. Do not commit API keys or private data to the repository.
4. Open the notebooks in Jupyter or run the relevant screening scripts using their expected inputs and configuration.


## Reproducibility notes

- Search results can change as source databases add or update records.
- Model-assisted screening results can vary with model versions.



