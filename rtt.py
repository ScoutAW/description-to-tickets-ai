"""
rtt.py  -  RAW TEXT -> TASKS
Stages 2, 3 and 4 of the flow: Understand, Detect gaps, Break down.

Design in one sentence: the LLM proposes, plain Python checks.
Every quote the LLM returns must exist word-for-word in the brief,
otherwise the item is dropped and reported as a warning.
Uses the Groq API (model openai/gpt-oss-20b). The key is read from the environment or a .env file
(GROQ_API_KEY, or OPENAI_API_KEY if that is the name you used).
"""
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()  # reads the .env file next to the code (keys never go in the source)

MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
NOT_SPECIFIED = "Not specified"
GAP_TYPES = {"missing", "vague", "contradiction", "undefined_term", "scope_gap"}

# --------------------------------------------------------------------------
# Prompts (one stage = one prompt, each with role, rules and exact output)
# --------------------------------------------------------------------------
GOLDEN_RULE = """GOLDEN RULE - NEVER ASSUME.
- Use ONLY what the brief literally says. Never fill a gap with a guess.
- Never add features, technologies, providers, priorities, estimates or deadlines the brief does not state.
- Every quote must be copied EXACTLY, character for character, from the brief. No ellipses, no paraphrase inside quotes.
- If the brief is silent, the value is "Not specified" and you raise a question instead.
- Reply with ONE valid JSON object and nothing else (no markdown fences, no commentary)."""

UNDERSTAND_SYSTEM = f"""ROLE: Requirements analyst. You extract what a client brief STATES. You do not interpret, improve or complete it.

{GOLDEN_RULE}

TASK: List every goal, feature, user group, constraint, integration, deadline and other stated fact.
Each item gets a short neutral statement and the exact supporting quote.

OUTPUT FORMAT:
{{"items": [{{"id": "E1", "category": "goal|feature|user|constraint|integration|deadline|other", "statement": "...", "quote": "exact text from brief"}}]}}"""

GAPS_SYSTEM = f"""ROLE: Skeptical reviewer who protects developers from guessing. You find what is missing, vague, contradictory or undefined in a client brief.

{GOLDEN_RULE}

Find ALL of these and turn each into ONE specific question the client can answer:
- missing: something a developer needs that the brief never states (target users, platform, integrations, deadlines, ...)
- vague: words with no measurable meaning ("fast", "modern", "user-friendly", "secure", ...). Quote the word in its sentence.
- contradiction: two parts that disagree. Provide BOTH quotes.
- undefined_term: names, systems or acronyms used without explanation.
- scope_gap: something a stated feature obviously needs but the brief does not mention. ASK about it; never add it.
Questions must be concrete (e.g. "Which payment methods and currencies are required?"), never generic ("Can you clarify?").
For vague, undefined_term and contradiction you MUST give exact quotes. For missing and scope_gap, quotes may be an empty list; use related_items to point at the extracted items concerned.

OUTPUT FORMAT:
{{"questions": [{{"type": "missing|vague|contradiction|undefined_term|scope_gap", "question": "...", "quotes": ["exact text"], "related_items": ["E1"]}}]}}"""

BREAKDOWN_SYSTEM = f"""ROLE: Technical project manager turning a brief into a developer-ready breakdown.

{GOLDEN_RULE}

RULES:
- Group work into features. Split each feature into SMALL tasks a developer can pick up. Create tasks only for things the brief states.
- description: what to build and why, in the words of the brief.
- acceptance_criteria: only testable conditions taken from the brief, each with its own exact quote. If the brief gives none, return an empty list. Never invent criteria.
- type: "Story" if the brief describes it from a user's point of view, otherwise "Task".
- depends_on: ONLY if the brief itself makes the link clear; give the exact quote. Otherwise leave empty.
- priority, estimate, deadline: fill ONLY if the brief states them (value + exact quote). Otherwise null.
- open_question_ids: IDs from the provided question list that must be answered before THIS task can start.
- Do not create tasks for gaps. Gaps stay as questions.

OUTPUT FORMAT:
{{"features": [{{"id": "F1", "title": "...", "description": "...", "source_quotes": ["exact"],
  "tasks": [{{"id": "T1", "title": "action-oriented title", "type": "Story|Task", "description": "...",
    "source_quotes": ["exact"],
    "acceptance_criteria": [{{"text": "...", "quote": "exact"}}],
    "depends_on": [{{"task": "T2", "quote": "exact"}}],
    "priority": {{"value": "...", "quote": "exact"}} or null,
    "estimate": {{"value": "...", "quote": "exact"}} or null,
    "deadline": {{"value": "...", "quote": "exact"}} or null,
    "open_question_ids": ["Q1"]}}]}}]}}"""

# --------------------------------------------------------------------------
# LLM call + JSON parsing
# --------------------------------------------------------------------------
_client = None


def get_api_key():
    return os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")


def _parse_json(text):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


def call_llm(system, user, max_tokens=8000):
    """Call the model on Groq, expect JSON, retry once if the reply is not valid JSON."""
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=get_api_key())
    note = ""
    for _ in range(2):
        kwargs = dict(
            model=MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user + note}],
            temperature=0,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},  # forces valid JSON
            reasoning_effort="low",                   # gpt-oss: keeps answers fast
        )
        try:
            resp = _client.chat.completions.create(**kwargs)
        except Exception:
            # some settings may be rejected; retry with only the essentials
            for k in ("reasoning_effort", "response_format"):
                kwargs.pop(k, None)
            resp = _client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        try:
            return _parse_json(text)
        except ValueError:
            note = "\n\nYour previous reply was not valid JSON. Reply with the JSON object only."
    raise RuntimeError("The model did not return valid JSON twice in a row.")


# --------------------------------------------------------------------------
# Checks (plain Python: this is the anti-hallucination layer)
# --------------------------------------------------------------------------
def _norm(s):
    s = (s or "").translate(str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"',
                                           "\u201d": '"', "\u2013": "-", "\u2014": "-"}))
    return re.sub(r"\s+", " ", s).strip().lower()


def quote_ok(quote, brief_norm):
    """True only if the quote appears word-for-word in the brief."""
    q = _norm(quote)
    return len(q) >= 3 and q in brief_norm


def _as_list(v):
    """Small models sometimes return a string where a list is expected."""
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _dicts(v, warnings, where):
    """Keep only well-formed (dict) entries; report anything else instead of crashing."""
    out = []
    for x in _as_list(v):
        if isinstance(x, dict):
            out.append(x)
        else:
            warnings.append(f"Ignored malformed entry in {where}: {str(x)[:80]!r}")
    return out


def _good_quotes(quotes, brief_norm):
    return [q for q in _as_list(quotes) if isinstance(q, str) and quote_ok(q, brief_norm)]


def _stated(field, brief_norm, label, warnings):
    """priority/estimate/deadline: keep only if value AND a verified quote exist."""
    if isinstance(field, dict) and field.get("value") and quote_ok(field.get("quote"), brief_norm):
        return str(field["value"])
    if field:
        warnings.append(f"Dropped unsupported {label}: {field}")
    return NOT_SPECIFIED


# --------------------------------------------------------------------------
# Stage 2: Understand
# --------------------------------------------------------------------------
def understand(brief, warnings):
    bn = _norm(brief)
    data = call_llm(UNDERSTAND_SYSTEM, f"BRIEF:\n\"\"\"\n{brief}\n\"\"\"")
    items = []
    for it in _dicts(data.get("items"), warnings, "items"):
        if quote_ok(it.get("quote"), bn):
            items.append({"id": f"E{len(items)+1}",
                          "category": str(it.get("category") or "other"),
                          "statement": str(it.get("statement") or ""),
                          "quote": it["quote"]})
        else:
            warnings.append(f"Dropped extracted item (quote not found in brief): {it.get('statement')!r}")
    return items


# --------------------------------------------------------------------------
# Stage 3: Detect gaps
# --------------------------------------------------------------------------
def detect_gaps(brief, items, warnings):
    bn = _norm(brief)
    user = (f"BRIEF:\n\"\"\"\n{brief}\n\"\"\"\n\nEXTRACTED ITEMS (already verified):\n"
            f"{json.dumps(items, indent=1)}")
    data = call_llm(GAPS_SYSTEM, user)
    ids = {i["id"] for i in items}
    questions = []
    for q in _dicts(data.get("questions"), warnings, "questions"):
        qtype = q.get("type")
        text = str(q.get("question") or "").strip()
        quotes = _good_quotes(q.get("quotes"), bn)
        if qtype not in GAP_TYPES or not text:
            warnings.append(f"Dropped malformed question: {str(q)[:120]}")
            continue
        need = {"contradiction": 2, "vague": 1, "undefined_term": 1}.get(qtype, 0)
        if len(quotes) < need:
            warnings.append(f"Dropped {qtype} question, needs {need} verified quote(s): {text!r}")
            continue
        questions.append({"id": f"Q{len(questions)+1}", "type": qtype, "question": text,
                          "quotes": quotes,
                          "related_items": [r for r in _as_list(q.get("related_items")) if r in ids]})
    return questions


# --------------------------------------------------------------------------
# Stage 4: Break down
# --------------------------------------------------------------------------
def break_down(brief, items, questions, warnings):
    bn = _norm(brief)
    user = (f"BRIEF:\n\"\"\"\n{brief}\n\"\"\"\n\nEXTRACTED ITEMS:\n{json.dumps(items, indent=1)}\n\n"
            f"OPEN QUESTIONS (use their ids in open_question_ids):\n{json.dumps(questions, indent=1)}")
    data = call_llm(BREAKDOWN_SYSTEM, user, max_tokens=12000)

    qids = {q["id"] for q in questions}
    raw_features = _dicts(data.get("features"), warnings, "features")
    for f in raw_features:
        f["tasks"] = _dicts(f.get("tasks"), warnings, "tasks")
    raw_tasks = {t.get("id"): t for f in raw_features for t in f["tasks"]
                 if isinstance(t.get("id"), str)}
    features, seen = [], set()

    def uid(value, prefix):
        """Keep the model's id if it is a unique string, otherwise make one."""
        if isinstance(value, str) and value and value not in seen:
            seen.add(value)
            return value
        n = len(seen) + 1
        while f"{prefix}{n}" in seen:
            n += 1
        seen.add(f"{prefix}{n}")
        return f"{prefix}{n}"

    for f in raw_features:
        tasks = []
        for t in f["tasks"]:
            title = str(t.get("title") or "").strip()
            quotes = _good_quotes(t.get("source_quotes"), bn)
            if not quotes or not title:
                warnings.append(f"Dropped task without a verified source quote: {title!r}")
                continue
            criteria = []
            for c in _as_list(t.get("acceptance_criteria")):
                if isinstance(c, dict) and c.get("text") and quote_ok(c.get("quote"), bn):
                    criteria.append({"text": str(c["text"]), "quote": c["quote"]})
                else:
                    warnings.append(f"Dropped acceptance criterion with no brief support in {title!r}: {str(c)[:100]}")
            tasks.append({
                "id": uid(t.get("id"), "T"), "title": title,
                "type": t.get("type") if t.get("type") in ("Story", "Task") else "Task",
                "description": str(t.get("description") or ""), "source_quotes": quotes,
                "acceptance_criteria": criteria, "dependencies": [],
                "priority": _stated(t.get("priority"), bn, "priority", warnings),
                "estimate": _stated(t.get("estimate"), bn, "estimate", warnings),
                "deadline": _stated(t.get("deadline"), bn, "deadline", warnings),
                "open_question_ids": [q for q in _as_list(t.get("open_question_ids"))
                                      if isinstance(q, str) and q in qids],
                "_dep_claims": _as_list(t.get("depends_on")),
                "_orig_id": t.get("id"),
            })
        fq = _good_quotes(f.get("source_quotes"), bn) or [q for t in tasks for q in t["source_quotes"]][:3]
        if tasks and fq:
            features.append({"id": uid(f.get("id"), "F"), "title": str(f.get("title") or "Untitled feature"),
                             "description": str(f.get("description") or ""),
                             "source_quotes": fq, "tasks": tasks})
        elif tasks:
            warnings.append(f"Dropped feature without source: {f.get('title')!r}")

    valid_ids = {t["id"] for f in features for t in f["tasks"]}
    orig_to_new = {t["_orig_id"]: t["id"] for f in features for t in f["tasks"]
                   if isinstance(t["_orig_id"], str)}
    extra_questions = []

    def new_q(qtype, text, quotes=()):
        qid = f"Q{len(questions) + len(extra_questions) + 1}"
        extra_questions.append({"id": qid, "type": qtype, "question": text,
                                "quotes": list(quotes), "related_items": []})
        return qid

    for f in features:
        for t in f["tasks"]:
            # dependencies: keep only if the brief states the link, else turn into a question
            for d in t.pop("_dep_claims"):
                if not isinstance(d, dict):
                    continue
                target = orig_to_new.get(d.get("task"))
                if not target or target == t["id"]:
                    continue
                if quote_ok(d.get("quote"), bn):
                    t["dependencies"].append(target)
                else:
                    other = raw_tasks.get(d.get("task"), {}).get("title", target)
                    t["open_question_ids"].append(new_q(
                        "missing", f"Must '{t['title']}' wait for '{other}'? The brief does not say."))
            t.pop("_orig_id", None)
            # no acceptance criteria in brief -> ask, never invent
            if not t["acceptance_criteria"]:
                t["open_question_ids"].append(new_q(
                    "missing", f"What conditions must be true for '{t['title']}' to be accepted? "
                               f"The brief gives no acceptance criteria.", t["source_quotes"][:1]))

    all_q = questions + extra_questions
    # every question must block at least one task
    for q in all_q:
        blocked = [t["id"] for f in features for t in f["tasks"] if q["id"] in t["open_question_ids"]]
        q["scope"] = "specific" if blocked else "project-wide"
        q["blocks"] = blocked or sorted(valid_ids)
        if not blocked:  # attach project-wide questions to every task so no one starts blind
            for f in features:
                for t in f["tasks"]:
                    t["open_question_ids"].append(q["id"])
    for f in features:
        for t in f["tasks"]:
            t["open_question_ids"] = sorted(set(t["open_question_ids"]), key=lambda x: int(x[1:]))
    return features, all_q


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_pipeline(brief, progress=None):
    """Brief text -> dict(items, questions, features, warnings). Each stage feeds the next."""
    brief = (brief or "").strip()
    if len(brief) < 20:
        raise ValueError("The brief is empty or too short to analyse.")
    say = progress or (lambda msg: None)
    warnings = []
    say("Stage 2/4: extracting what the brief states...")
    items = understand(brief, warnings)
    say("Stage 3/4: looking for gaps, vague wording and contradictions...")
    questions = detect_gaps(brief, items, warnings)
    say("Stage 4/4: breaking the work into features and tasks...")
    features, questions = break_down(brief, items, questions, warnings)
    return {"brief": brief, "items": items, "questions": questions,
            "features": features, "warnings": warnings}


if __name__ == "__main__":
    import sys
    import tr
    path = sys.argv[1]
    result = run_pipeline(open(path, encoding="utf-8").read(), progress=print)
    package = tr.build_package(result)
    stem = os.path.splitext(os.path.basename(path))[0]
    os.makedirs("output", exist_ok=True)
    tr.write_exports(package, os.path.join("output", stem))
    print(f"Done. Files written to output/{stem}.*")
