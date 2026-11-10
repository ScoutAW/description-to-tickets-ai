"""
ui.py  -  USER INTERFACE (Stage 1 Input and Stage 6 Output)
Run with:  streamlit run ui.py
No login, no accounts. Paste a brief or upload a .txt file.
"""
import os

import streamlit as st

import rtt
import tr

st.set_page_config(page_title="Brief to tickets", layout="wide")
st.title("Client brief to developer tickets")
st.write("Paste a client brief. The tool extracts only what it states, lists what is "
         "missing or unclear as questions, and creates tickets. It never fills gaps with guesses.")

if not rtt.get_api_key():
    st.error("No API key found. Add GROQ_API_KEY=... to the .env file next to ui.py, then restart the app.")
    st.stop()

uploaded = st.file_uploader("Load a brief from a text file (optional)", type=["txt", "md"])
default = uploaded.read().decode("utf-8", errors="replace") if uploaded else ""
brief = st.text_area("Client brief", value=default, height=260)

if st.button("Analyse brief", type="primary"):
    try:
        with st.status("Working...", expanded=True) as status:
            result = rtt.run_pipeline(brief, progress=st.write)
            st.session_state["package"] = tr.build_package(result)
            status.update(label="Done", state="complete")
    except Exception as e:  # show the real reason to the user
        st.error(f"Could not analyse the brief: {e}")

pkg = st.session_state.get("package")
if pkg:
    tickets, questions = pkg["tickets"], pkg["open_questions"]
    ready = sum(t["status"] == tr.READY for t in tickets)
    c1, c2, c3 = st.columns(3)
    c1.metric("Tickets", len(tickets))
    c2.metric("Ready for development", ready)
    c3.metric("Open questions", len(questions))

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download JSON", tr.to_json(pkg), "tickets.json", "application/json")
    d2.download_button("Download CSV (Jira import)", tr.to_csv(pkg), "tickets.csv", "text/csv")
    d3.download_button("Download full report", tr.to_markdown(pkg), "report.md", "text/markdown")

    t_open, t_tickets, t_extract = st.tabs(["Open questions", "Breakdown and tickets", "What the brief states"])

    with t_open:
        for q in pkg["open_questions"]:
            scope = "all tickets" if q["scope"] == "project-wide" else ", ".join(q["blocks_tickets"])
            st.markdown(f'**{q["id"]}** ({q["type"].replace("_", " ")}): {q["question"]}')
            for s in q["brief_quotes"]:
                st.caption(f'Brief: "{s}"')
            st.caption(f"Blocks: {scope}")
            st.divider()

    with t_tickets:
        for t in tickets:
            flag = "Ready" if t["status"] == tr.READY else "Needs clarification"
            indent = "" if t["type"] == "Epic" else "    "
            with st.expander(f'{indent}{t["ticket_id"]}  {t["title"]}  [{t["type"]}, {flag}]'):
                st.write(t["description"])
                st.write(f'Priority: {t["priority"]} | Estimate: {t["estimate"]} | Deadline: {t["deadline"]}')
                st.write("Depends on: " + (", ".join(t["dependencies"]) or "None stated in brief"))
                if t["acceptance_criteria"]:
                    st.write("Acceptance criteria")
                    for c in t["acceptance_criteria"]:
                        st.write(f"- {c}")
                st.write("Source in brief")
                for s in t["source_reference"]:
                    st.caption(f'"{s}"')
                if t["open_questions"]:
                    st.write("Open questions")
                    for q in t["open_questions"]:
                        st.write(f'- {q["id"]}: {q["question"]}')

    with t_extract:
        for i in pkg["extracted"]:
            st.markdown(f'**{i["category"]}**: {i["statement"]}')
            st.caption(f'"{i["quote"]}"')

    if pkg["warnings"]:
        with st.expander(f'Validation notes ({len(pkg["warnings"])} items removed because the brief did not support them)'):
            for w in pkg["warnings"]:
                st.write(f"- {w}")
