import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel

from knowledge_base.loader import (
    load_all_articles,
    save_article,
    delete_article,
    extract_text,
)

load_dotenv()

# ---------------------------------------------------------------- Gemini setup
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Put GEMINI_MODEL=<current Flash / Flash-Lite model> in .env
# (see https://ai.google.dev/gemini-api/docs/models)
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# If you get an error about thinking_budget, delete the thinking_config line.
FAST = dict(
    temperature=0.1,
    thinking_config=types.ThinkingConfig(thinking_budget=0),
)

LOG_FILE = Path("outputs/incident_log.csv")


class Resolution(BaseModel):
    matched_article: str  # article name from the KB, or "NONE"
    confidence: Literal["High", "Medium", "Low"]
    summary: str
    steps: list[str]
    escalate: bool
    support_team: str
    reason: str


SYSTEM_PROMPT = """You are an IT support agent. A user reports an incident.
Search ONLY the knowledge base articles provided and pick the single best match.
Rules:
- Use only information that is in the matched article. Never invent steps.
- matched_article must be exactly one of the article names, or "NONE" if nothing fits.
- If NONE: steps = [], escalate = true, support_team = "Service Desk", and explain why in reason.
- If matched: copy the resolution steps from the article (short, numbered order),
  set escalate = true only if the article's escalation criteria clearly apply
  to this incident, and set support_team from the article's support team.
- confidence: High = symptoms clearly match, Medium = partial match, Low = weak match.
- Keep summary to 1-2 sentences and reason to 1 sentence."""


# --------------------------------------------------------------------- helpers
def shrink(img: Image.Image, max_side: int = 1024) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def image_to_article_text(img: Image.Image) -> str:
    """Transcribe a screenshot of a KB article into markdown (done once, then saved)."""
    resp = client.models.generate_content(
        model=MODEL,
        contents=[
            "Transcribe this IT knowledge article into clean markdown. Keep the title, "
            "symptoms, troubleshooting steps, escalation criteria and support team. "
            "Output only the markdown.",
            shrink(img, 1600),
        ],
        config=types.GenerateContentConfig(max_output_tokens=1500, **FAST),
    )
    return resp.text or ""


def find_resolution(incident_text: str, screenshot, articles: dict) -> Resolution:
    kb = "\n\n".join(f"### ARTICLE: {n}\n{t}" for n, t in articles.items())
    contents = [
        f"KNOWLEDGE BASE:\n{kb}\n\nINCIDENT:\n{incident_text or '(see screenshot)'}"
    ]
    if screenshot is not None:
        contents.append(shrink(screenshot))

    resp = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=Resolution,
            max_output_tokens=800,
            **FAST,
        ),
    )
    result = resp.parsed or Resolution(**json.loads(resp.text))
    # Guard against a made-up article name
    if result.matched_article != "NONE" and result.matched_article not in articles:
        result.matched_article = "NONE"
        result.escalate = True
    return result


def log_incident(status: str):
    last = st.session_state.get("last")
    if not last:
        return
    LOG_FILE.parent.mkdir(exist_ok=True)
    new_file = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "incident", "matched_article", "confidence", "status"])
        w.writerow([
            datetime.now().isoformat(timespec="seconds"),
            last["incident"].replace("\n", " "),
            last["result"]["matched_article"],
            last["result"]["confidence"],
            status,
        ])
    st.session_state["status_msg"] = (
        "✅ Marked as resolved and logged (ticket would be closed)."
        if status == "Resolved"
        else "⬆️ Escalated and logged."
    )


# ------------------------------------------------------------------------- UI
st.set_page_config(page_title="AI Support Assistant", page_icon="🤖", layout="wide")
st.title("🤖 AI Support Assistant")

articles = load_all_articles()

# ---- Sidebar: knowledge base (saved on disk, remembered between runs) ----
with st.sidebar:
    st.header("📚 Knowledge Base")
    st.caption(f"{len(articles)} article(s) loaded")

    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))

    st.session_state.setdefault("up_key", 0)
    new_files = st.file_uploader(
        "Add articles (saved permanently)",
        type=["md", "txt", "html", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=f"kb_up_{st.session_state['up_key']}",
    )
    if st.button("Save to knowledge base", disabled=not new_files):
        saved = []
        with st.spinner("Saving..."):
            for f in new_files:
                ext = Path(f.name).suffix.lower()
                if ext in {".png", ".jpg", ".jpeg"}:
                    text = image_to_article_text(Image.open(f))
                else:
                    text = extract_text(f.name, f.getvalue())
                saved.append(save_article(f.name, text))
        st.session_state["flash"] = "Saved: " + ", ".join(saved)
        st.session_state["up_key"] += 1  # clears the uploader
        st.rerun()

    if articles:
        with st.expander("View / remove articles"):
            for name in articles:
                st.write(f"• {name}")
            to_delete = st.selectbox("Remove an article", [""] + list(articles))
            if st.button("Delete", disabled=not to_delete):
                delete_article(to_delete)
                st.rerun()

# ---- Main: report an incident ----
st.subheader("Describe the incident")
incident = st.text_area(
    "What is the user's problem?",
    placeholder="e.g. User cannot connect to VPN, getting an authentication error after entering MFA code",
    height=120,
)
screenshot_file = st.file_uploader(
    "Optional: screenshot of the error", type=["png", "jpg", "jpeg"], key="shot"
)

if st.button("🔍 Find resolution", type="primary"):
    if not articles:
        st.error("The knowledge base is empty. Add articles in the sidebar first.")
    elif not incident.strip() and screenshot_file is None:
        st.warning("Please describe the incident or attach a screenshot.")
    else:
        shot = Image.open(screenshot_file) if screenshot_file else None
        with st.spinner("Searching knowledge base..."):
            result = find_resolution(incident, shot, articles)
        st.session_state["last"] = {"incident": incident, "result": result.model_dump()}
        st.session_state.pop("status_msg", None)

# ---- Show result (kept in session so button clicks don't lose it) ----
last = st.session_state.get("last")
if last:
    r = Resolution(**last["result"])
    st.divider()

    if r.matched_article == "NONE":
        st.warning("No matching article found in the knowledge base.")
        st.write(r.reason)
    else:
        st.success(f"Matched article: **{r.matched_article}**  |  Confidence: **{r.confidence}**")
        st.write(r.summary)
        st.markdown("**Resolution steps**")
        for i, step in enumerate(r.steps, 1):
            st.markdown(f"{i}. {step}")
        st.caption(r.reason)
        with st.expander("View source article"):
            st.markdown(articles.get(r.matched_article, ""))

    if r.escalate:
        st.info(f"If these steps don't work, escalate to: **{r.support_team}**")

    if "status_msg" in st.session_state:
        st.success(st.session_state["status_msg"])
    else:
        c1, c2, _ = st.columns([1, 1, 3])
        c1.button("✅ Resolved – close ticket", on_click=log_incident, args=("Resolved",))
        c2.button("⬆️ Not resolved – escalate", on_click=log_incident, args=("Escalated",))