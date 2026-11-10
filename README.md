# Client Brief → Dev-Ready Tickets

AI tool that converts a raw client brief into a structured task breakdown,
Jira-style tickets, and a list of open questions for the client.

## Libraries used
- Python 3.x
- Streamlit (UI)
- Groq API (LLM)
- python-dotenv (environment variables)
(adjust to match requirements.txt)

## How to run
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your `GROQ_API_KEY`
3. `streamlit run ui.py`
No login or accounts needed. Paste a brief or upload a .txt file.

## How it works (stages)
1. Input → 2. Understand → 3. Detect gaps → 4. Break down → 5. Generate tickets → 6. Output
(one prompt per stage; the checked output of each stage feeds the next)

## How it avoids assumptions
- Every prompt states the golden rule: if the brief is silent, write "Not specified" and raise a question
- Every extracted item must quote the brief
- Gaps, vague terms, contradictions, undefined terms and scope gaps become specific client questions linked to the tickets they block
- Priority, estimate and deadline are only filled if the brief states them
- Dependencies are only listed when the brief makes the link clear

## Security
API key is read from the `.env` file, which is git-ignored.