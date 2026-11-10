"""
tr.py  -  TICKETS & REPORT
Stages 5 and 6 of the flow: Generate tickets, Output.

"""
import csv
import io
import json

NOT_SPECIFIED = "Not specified"
READY = "Ready for development"
NEEDS = "Needs clarification"

CSV_COLUMNS = ["Issue ID", "Summary", "Issue Type", "Description", "Acceptance Criteria",
               "Source Reference", "Depends On", "Open Questions", "Status",
               "Priority", "Estimate", "Due Date", "Epic Link"]


def build_package(result):
    """Turn the task breakdown into tickets + open questions with consistent IDs."""
    ticket_of = {}   # internal id (F1/T1) -> TASK-001
    counter = 0
    for f in result["features"]:
        counter += 1
        ticket_of[f["id"]] = f"TASK-{counter:03d}"
        for t in f["tasks"]:
            counter += 1
            ticket_of[t["id"]] = f"TASK-{counter:03d}"

    qmap = {q["id"]: q for q in result["questions"]}
    tickets = []
    for f in result["features"]:
        epic_id = ticket_of[f["id"]]
        child_q = sorted({qid for t in f["tasks"] for qid in t["open_question_ids"]},
                         key=lambda x: int(x[1:]))
        tickets.append(_ticket(epic_id, f["title"], "Epic", f["description"], [],
                               f["source_quotes"], [], child_q, qmap,
                               NOT_SPECIFIED, NOT_SPECIFIED, NOT_SPECIFIED, None))
        for t in f["tasks"]:
            tickets.append(_ticket(
                ticket_of[t["id"]], t["title"], t["type"], t["description"],
                [c["text"] for c in t["acceptance_criteria"]], t["source_quotes"],
                [ticket_of[d] for d in t["dependencies"]], t["open_question_ids"], qmap,
                t["priority"], t["estimate"], t["deadline"], epic_id))

    questions = [{"id": q["id"], "type": q["type"], "question": q["question"],
                  "brief_quotes": q["quotes"], "scope": q["scope"],
                  "blocks_tickets": [ticket_of[b] for b in q["blocks"]]}
                 for q in result["questions"]]
    return {"extracted": result["items"], "tickets": tickets,
            "open_questions": questions, "warnings": result["warnings"]}


def _ticket(tid, title, ttype, desc, criteria, sources, deps, qids, qmap,
            priority, estimate, deadline, epic):
    return {
        "ticket_id": tid, "title": title, "type": ttype, "description": desc,
        "acceptance_criteria": criteria, "source_reference": sources,
        "dependencies": deps,
        "open_questions": [{"id": q, "question": qmap[q]["question"]} for q in qids],
        "status": NEEDS if qids else READY,
        "priority": priority or NOT_SPECIFIED, "estimate": estimate or NOT_SPECIFIED,
        "deadline": deadline or NOT_SPECIFIED, "epic": epic,
    }


# ---------------------------------------------------------------- exports
def to_json(package):
    return json.dumps(package, indent=2, ensure_ascii=False)


def to_csv(package):
    """Jira-style CSV (import via Jira's CSV importer)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
    w.writeheader()
    for t in package["tickets"]:
        w.writerow({
            "Issue ID": t["ticket_id"], "Summary": t["title"], "Issue Type": t["type"],
            "Description": t["description"],
            "Acceptance Criteria": "\n".join(f"- {c}" for c in t["acceptance_criteria"]),
            "Source Reference": "\n".join(f'"{s}"' for s in t["source_reference"]),
            "Depends On": ", ".join(t["dependencies"]) or "None stated in brief",
            "Open Questions": "\n".join(f'{q["id"]}: {q["question"]}' for q in t["open_questions"]),
            "Status": t["status"], "Priority": t["priority"], "Estimate": t["estimate"],
            "Due Date": t["deadline"], "Epic Link": t["epic"] or "",
        })
    return buf.getvalue()


def to_markdown(package):
    L = ["# Client brief analysis", ""]
    L += ["## 1. What the brief states", ""]
    for i in package["extracted"]:
        L.append(f'- **{i["category"]}**: {i["statement"]}  \n  > "{i["quote"]}"')
    L += ["", "## 2. Open questions for the client", ""]
    for q in package["open_questions"]:
        L.append(f'- **{q["id"]}** ({q["type"]}): {q["question"]}')
        for s in q["brief_quotes"]:
            L.append(f'  > "{s}"')
        scope = "all tickets" if q["scope"] == "project-wide" else ", ".join(q["blocks_tickets"])
        L.append(f"  Blocks: {scope}")
    L += ["", "## 3. Breakdown and tickets", ""]
    for t in package["tickets"]:
        L += [f'### {t["ticket_id"]} - {t["title"]}',
              f'- Type: {t["type"]}  |  Status: **{t["status"]}**',
              f'- Priority: {t["priority"]}  |  Estimate: {t["estimate"]}  |  Deadline: {t["deadline"]}',
              f'- Epic: {t["epic"] or "-"}  |  Depends on: {", ".join(t["dependencies"]) or "None stated in brief"}',
              f'- Description: {t["description"]}']
        if t["acceptance_criteria"]:
            L.append("- Acceptance criteria:")
            L += [f"  - {c}" for c in t["acceptance_criteria"]]
        else:
            L.append("- Acceptance criteria: none stated in brief")
        L.append("- Source:")
        L += [f'  > "{s}"' for s in t["source_reference"]]
        if t["open_questions"]:
            L.append("- Open questions:")
            L += [f'  - {q["id"]}: {q["question"]}' for q in t["open_questions"]]
        L.append("")
    if package["warnings"]:
        L += ["## 4. Validation notes (items removed because the brief did not support them)", ""]
        L += [f"- {w}" for w in package["warnings"]]
    return "\n".join(L)


def write_exports(package, stem):
    for ext, fn in (("json", to_json), ("csv", to_csv), ("md", to_markdown)):
        with open(f"{stem}.{ext}", "w", encoding="utf-8", newline="") as f:
            f.write(fn(package))
