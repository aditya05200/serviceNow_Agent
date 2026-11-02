import csv
import os
import re
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

from knowledge_base.loader import (
    load_all_articles,
    save_article,
    delete_article,
    extract_text,
)

load_dotenv()

# ---------------------------------------------------------------- Gemini setup
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# A GEMINI_MODEL value in .env overrides this default.
# Flash-Lite models are the fastest and are enough for matching a ticket to a KB article.
# To see which model names your key can use, run: [m.name for m in client.models.list()]
MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")

LOG_FILE = Path("outputs/incident_log.csv")


def _thinking_options():
    """Thinking settings to try, best first. The last entry (None) = model default.
    If a model rejects a setting, we automatically fall back to the next one."""
    m = MODEL.lower()
    opts = []
    try:
        if "2.5" in m:
            opts.append(types.ThinkingConfig(thinking_budget=0))
        elif re.search(r"gemini-([3-9])", m):
            opts.append(types.ThinkingConfig(thinking_level="low"))
    except Exception:
        pass  # installed SDK doesn't know this setting
    opts.append(None)
    return opts


THINK_OPTS = _thinking_options()


def _config(thinking, **kw):
    if thinking is not None:
        kw["thinking_config"] = thinking
    return types.GenerateContentConfig(temperature=0.1, **kw)


def generate(contents, **kw):
    """Normal (non-streaming) Gemini call with thinking fallback."""
    err = None
    for i in range(st.session_state.get("think_idx", 0), len(THINK_OPTS)):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=contents, config=_config(THINK_OPTS[i], **kw)
            )
            st.session_state["think_idx"] = i
            return resp
        except Exception as e:
            if "thinking" not in str(e).lower():
                raise
            err = e
    raise err


def generate_stream(contents, **kw):
    """Streaming Gemini call with thinking fallback. Yields text pieces."""
    err = None
    for i in range(st.session_state.get("think_idx", 0), len(THINK_OPTS)):
        started = False
        try:
            for chunk in client.models.generate_content_stream(
                model=MODEL, contents=contents, config=_config(THINK_OPTS[i], **kw)
            ):
                started = True
                if chunk.text:
                    yield chunk.text
            st.session_state["think_idx"] = i
            return
        except Exception as e:
            if started or "thinking" not in str(e).lower():
                raise
            err = e
    raise err


SYSTEM_PROMPT = """You are an IT support agent. A user reports an incident.
Search ONLY the knowledge base articles provided and pick the single best match.
Rules:
- Use only information that is in the matched article. Never invent steps.
- MATCH must be exactly one of the article names, or NONE if nothing fits.
- Copy the resolution steps from the article (short, in order).
- ESCALATE = yes only if the article's escalation criteria clearly apply to this
  incident, or if MATCH is NONE. TEAM = the article's support team, or Service Desk.
- CONFIDENCE: High = symptoms clearly match, Medium = partial match, Low = weak match.

Reply in EXACTLY this format, with nothing before the first line:
MATCH: <exact article name or NONE>
CONFIDENCE: <High|Medium|Low>
ESCALATE: <yes|no>
TEAM: <support team>
---
**Summary:** <1-2 sentences>

**Resolution steps:**
1. <step>
2. <step>

**Why this article:** <1 sentence>

If MATCH is NONE, replace the steps with one sentence explaining why nothing fits."""


# --------------------------------------------------------------------- helpers
def shrink(img: Image.Image, max_side: int = 1024) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    return img


def image_to_article_text(img: Image.Image) -> str:
    """Transcribe a screenshot of a KB article into markdown (done once, then saved)."""
    resp = generate(
        [
            "Transcribe this IT knowledge article into clean markdown. Keep the title, "
            "symptoms, troubleshooting steps, escalation criteria and support team. "
            "Output only the markdown.",
            shrink(img, 1600),
        ],
        max_output_tokens=2000,
    )
    return resp.text or ""


def parse_header(head: str, articles: dict) -> dict:
    """Read the MATCH / CONFIDENCE / ESCALATE / TEAM lines from the model reply."""

    def field(name, default=""):
        m = re.search(rf"^\W*{name}\W*:\s*(.+)$", head, re.M | re.I)
        return m.group(1).strip().strip("*`") if m else default

    lookup = {k.lower(): k for k in articles}
    key = field("MATCH", "NONE").lower().removesuffix(".md").strip()
    match = lookup.get(key, "NONE")

    conf = field("CONFIDENCE", "Low").capitalize()
    if conf not in ("High", "Medium", "Low"):
        conf = "Low"

    escalate = field("ESCALATE", "yes").lower().startswith("y") or match == "NONE"
    return dict(
        matched_article=match,
        confidence=conf,
        escalate=escalate,
        support_team=field("TEAM", "Service Desk") or "Service Desk",
    )


def stream_body(contents, raw: list, meta: dict, articles: dict, banner):
    """Stream the answer. The header lines are hidden from the stream and shown as a
    banner as soon as they are complete; the body is streamed word by word."""
    buf, in_body, started = "", False, False
    for text in generate_stream(
        contents, system_instruction=SYSTEM_PROMPT, max_output_tokens=1500
    ):
        meta.setdefault("first", time.perf_counter() - meta["t0"])
        raw.append(text)
        if not in_body:
            buf += text
            if "---" not in buf:
                continue
            in_body = True
            head, _, text = buf.partition("---")
            show_banner(banner, parse_header(head, articles))
        if not started:
            text = text.lstrip()  # drop blank space right after the --- line
            if not text:
                continue
            started = True
        yield text
    if not in_body and buf:  # model ignored the format: show everything
        yield buf


def show_banner(target, r: dict):
    if r["matched_article"] == "NONE":
        target.warning("No matching article found in the knowledge base.")
    else:
        target.success(
            f"Matched article: **{r['matched_article']}**  |  Confidence: **{r['confidence']}**"
        )


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
    st.caption(f"{len(articles)} article(s) loaded  ·  model: `{MODEL}`")

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

fresh = False  # True when the answer was just streamed on this run

if st.button("🔍 Find resolution", type="primary"):
    if not articles:
        st.error("The knowledge base is empty. Add articles in the sidebar first.")
    elif not incident.strip() and screenshot_file is None:
        st.warning("Please describe the incident or attach a screenshot.")
    else:
        kb = "\n\n".join(f"### ARTICLE: {n}\n{t}" for n, t in articles.items())
        contents = [
            f"KNOWLEDGE BASE:\n{kb}\n\nINCIDENT:\n{incident.strip() or '(see screenshot)'}"
        ]
        if screenshot_file is not None:
            contents.append(shrink(Image.open(screenshot_file)))

        st.session_state.pop("status_msg", None)
        st.session_state.pop("last", None)
        st.divider()
        banner = st.empty()
        raw, meta = [], {"t0": time.perf_counter()}
        try:
            with st.spinner("Searching knowledge base..."):
                # the spinner disappears as soon as the first words arrive
                body_box = st.container()
                body_box.write_stream(stream_body(contents, raw, meta, articles, banner))
            full = "".join(raw)
            head, sep, body = full.partition("---")
            if not sep:
                head, body = full, full
            result = parse_header(head, articles)
            result["body"] = body.strip()
            result["first"] = round(meta.get("first", 0), 1)
            result["total"] = round(time.perf_counter() - meta["t0"], 1)
            st.session_state["last"] = {"incident": incident, "result": result}
            fresh = True
        except Exception as e:
            st.error(f"Gemini request failed: {e}")
            st.caption(
                "If this mentions 404 / model not found, set GEMINI_MODEL in .env to a "
                "model name your key supports."
            )

# ---- Show result (kept in session so button clicks don't lose it) ----
last = st.session_state.get("last")
if last:
    r = last["result"]

    if not fresh:  # on the run that just streamed, this is already on screen
        st.divider()
        show_banner(st, r)
        st.markdown(r["body"])

    st.caption(f"⏱ first words after {r['first']}s · finished in {r['total']}s · model: {MODEL}")

    if r["matched_article"] != "NONE":
        with st.expander("View source article"):
            st.markdown(articles.get(r["matched_article"], ""))

    if r["escalate"]:
        st.info(f"If these steps don't work, escalate to: **{r['support_team']}**")

    if "status_msg" in st.session_state:
        st.success(st.session_state["status_msg"])
    else:
        c1, c2, _ = st.columns([1, 1, 3])
        c1.button("✅ Resolved – close ticket", on_click=log_incident, args=("Resolved",))
        c2.button("⬆️ Not resolved – escalate", on_click=log_incident, args=("Escalated",))