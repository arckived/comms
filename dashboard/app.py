"""Streamlit dashboard for the Comms Triage API.

    streamlit run dashboard/app.py          (API_URL defaults to http://localhost:8000)
"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TEAMS = ["PR", "Social", "Design", "Internal Comms", "Exec Comms"]
EXAMPLES = [
    "A reporter from The Verge wants a comment on the outage by 3pm today",
    "Can design make a banner for the launch blog post by Wednesday?",
    "yo can some1 post about the hackathon winners on linkedin",
    "Need talking points for the CEO's investor dinner on Thursday",
    "Please add the benefits enrollment reminder to this week's newsletter",
    "hey quick question",
]

st.set_page_config(page_title="Comms Triage", page_icon="📨", layout="wide")


@st.cache_resource
def client() -> httpx.Client:
    return httpx.Client(base_url=API_URL, timeout=60)


def api(method: str, path: str, **kwargs):
    try:
        resp = client().request(method, path, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text
    except httpx.HTTPError as exc:
        st.error(f"API error on {path}: {exc}")
        st.stop()


def show_request(req: dict):
    badge = "✅ Routed automatically" if req["route"] == "auto" else "🕵️ Sent to review"
    st.markdown(f"**{badge}** to **{req['final_team'] or req['predicted_team']}** "
                f"({req['confidence']:.0%} confident, threshold {req['threshold']:.0%})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Urgency", req["urgency"])
    c2.metric("Deadline", req["deadline"] or "none", req.get("deadline_date") or None, delta_color="off")
    c3.metric("Extraction", req["extraction_method"])
    if req["assets"]:
        st.write("**Needs:** " + ", ".join(req["assets"]))
    st.write("**Suggested reply:** " + req["draft_reply"])
    probs = pd.DataFrame({"team": list(req["probabilities"]), "probability": list(req["probabilities"].values())})
    st.bar_chart(probs.set_index("team"), horizontal=True, height=200)
    if req["related"]:
        st.write("**Related past work**")
        for r in req["related"]:
            title = f"[{r['title']}]({r['url']})" if r.get("url") else r["title"]
            kind = "Past request" if r["kind"] == "past_request" else "Announcement"
            st.markdown(f"- {kind}: {title} (similarity {r['score']:.2f})")


health = api("GET", "/health")
st.title("📨 Comms Triage")
st.caption(f"Model **{health['model']}** · embeddings **{health['backend']}** · threshold "
           f"**{health['threshold']}** · LLM **{'on' if health['llm_enabled'] else 'off (rules)'}**")

tab_submit, tab_review, tab_dash, tab_model, tab_search = st.tabs(
    ["Submit a request", "Review queue", "Dashboard", "Model", "Search"])

with tab_submit:
    example = st.selectbox("Try an example (optional)", [""] + EXAMPLES)
    with st.form("submit"):
        text = st.text_area("What do you need from the comms team?", value=example, height=120)
        requester = st.text_input("Your name (optional)")
        sent = st.form_submit_button("Submit", type="primary")
    if sent and text.strip():
        req = api("POST", "/triage", json={"text": text, "source": "web", "requester": requester or None})
        show_request(req)

with tab_review:
    pending = api("GET", "/requests", params={"status": "pending_review", "limit": 200})
    st.subheader(f"{len(pending)} waiting for a person")
    for req in pending:
        with st.container(border=True):
            st.markdown(f"**#{req['id']}** · {req['created_at'][:16].replace('T', ' ')} · via {req['source']}")
            st.markdown(f"> {req['text']}")
            top = ", ".join(f"{t} {p:.0%}" for t, p in list(req["probabilities"].items())[:3])
            st.caption(f"Model guess: {req['predicted_team']} ({top}) · urgency {req['urgency']}"
                       + (f" · deadline {req['deadline']}" if req["deadline"] else ""))
            guess = req["predicted_team"] if req["predicted_team"] in TEAMS else TEAMS[0]
            c1, c2 = st.columns([3, 1])
            team = c1.selectbox("Assign to", TEAMS, index=TEAMS.index(guess), key=f"team_{req['id']}")
            if c2.button("Confirm", key=f"ok_{req['id']}", type="primary"):
                api("POST", f"/requests/{req['id']}/review", json={"final_team": team, "reviewer": "dashboard"})
                st.rerun()

with tab_dash:
    stats = api("GET", "/stats")
    if not stats["total"]:
        st.info("No requests yet. Submit one, or run `python scripts/simulate.py` to load demo traffic.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Requests", stats["total"])
        c2.metric("Routed automatically", f"{stats['auto_rate']:.0%}")
        c3.metric("Waiting for review", stats["pending_review"])
        mins = stats["median_minutes_to_review"]
        c4.metric("Median minutes to review", "n/a" if mins is None else mins)
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Requests by team**")
            st.bar_chart(pd.Series(stats["by_team"], name="requests"))
        with c2:
            st.write("**Requests by urgency**")
            st.bar_chart(pd.Series(stats["by_urgency"], name="requests"))
        st.caption(f"Reviews that changed the model's team: {stats['review_changed_team']} of "
                   f"{stats['reviewed']}. Auto routed requests later overridden: "
                   f"{stats['auto_routed_then_overridden']}.")
        rows = api("GET", "/requests", params={"limit": 200})
        df = pd.DataFrame(rows)[["id", "created_at", "source", "text", "predicted_team", "confidence", "route",
                                 "status", "final_team", "urgency", "deadline"]]
        st.dataframe(df, hide_index=True, use_container_width=True)

with tab_model:
    info = api("GET", "/model")
    m = info.get("metrics")
    if not m:
        st.info("No metrics.json found for the loaded model.")
    else:
        t = m["test"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy", f"{t['accuracy']:.0%}")
        c2.metric("Test macro F1", f"{t['macro_f1']:.2f}")
        c3.metric("Auto routed (test)", f"{t['auto_rate']:.0%}")
        c4.metric("Accuracy when auto routed", "n/a" if t["auto_accuracy"] is None else f"{t['auto_accuracy']:.0%}")
        st.caption(f"{m['n_test']} handwritten test requests · {m['n_training_rows']} training rows · "
                   f"best of {', '.join(m['cv'])} by grouped CV · backend {m['backend']}")
        st.write("**Cross validation (grouped by seed request)**")
        st.dataframe(pd.DataFrame(m["cv"]).T, use_container_width=True)
        st.write("**Choosing the threshold** (out of fold predictions)")
        curve = pd.DataFrame(m["threshold_curve_oof"]).set_index("threshold")[["auto_rate", "auto_accuracy"]]
        st.line_chart(curve)
        cm = m["test"]["confusion_matrix"]
        st.write("**Confusion matrix (test)**: rows are the true team, columns the prediction")
        st.dataframe(pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"]),
                     use_container_width=True)
    for key, title in (("robustness", "Robustness"), ("retrieval", "Retrieval"),
                       ("extraction_rules", "Field extraction (rules)"), ("extraction_llm", "Field extraction (LLM)")):
        if key in info:
            st.write(f"**{title}**")
            data = {k: v for k, v in info[key].items() if not k.startswith("_") and k not in ("not_first", "deadline_misses")}
            if key == "robustness":
                st.dataframe(pd.DataFrame(data).T, use_container_width=True)
            else:
                st.json(data, expanded=False)

with tab_search:
    q = st.text_input("Search announcements and past requests", "When did we launch Edge Shield?")
    if q:
        for r in api("GET", "/search", params={"q": q, "k": 5}):
            title = f"[{r['title']}]({r['url']})" if r.get("url") else r["title"]
            st.markdown(f"**{title}** · {r['kind']} · score {r['score']:.2f}")
            st.caption(r["snippet"])
