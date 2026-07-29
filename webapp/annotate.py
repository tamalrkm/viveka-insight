"""Concept-precision annotation UI (paper evaluation, human study).

A deliberately tiny, model-free Streamlit app: annotators judge whether an
extracted concept correctly describes its paragraph. Two (or more) people
each work under their own user ID; progress is saved after every click, and
re-entering the same ID resumes where they left off.

Run:
    bash run_annotate.sh          # streamlit run webapp/annotate.py --server.port 8502

Data:
    items       docs/paper/eval/concept_sample_annotator_A.csv
                (created by `python scripts/eval_concept_precision.py sample`;
                 the _A/_B sheets are identical, we read A)
    judgments   docs/paper/eval/annotations/<userid>.json

When two users have finished all items:
    python scripts/eval_concept_precision.py compute
"""
from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "docs" / "paper" / "eval"
ITEMS_CSV = EVAL_DIR / "concept_sample_annotator_A.csv"
ANN_DIR = EVAL_DIR / "annotations"

_USERID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,31}$")

st.set_page_config(page_title="Concept annotation — viveka-insight",
                   page_icon="✍️", layout="centered")


# ──────────────────────────────────────────────────────────────────────────
# Data access
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_items() -> list[dict]:
    with ITEMS_CSV.open(encoding="utf-8") as f:
        items = list(csv.DictReader(f))
    items.sort(key=lambda r: int(r["item_id"]))
    return items


def _user_file(userid: str) -> Path:
    return ANN_DIR / f"{userid}.json"


def load_user(userid: str) -> dict:
    p = _user_file(userid)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"userid": userid, "judgments": {}}


def save_user(data: dict) -> None:
    """Atomic write so a mid-write refresh can't corrupt the file."""
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    p = _user_file(data["userid"])
    fd, tmp = tempfile.mkstemp(dir=str(ANN_DIR), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def known_users() -> list[tuple[str, int]]:
    if not ANN_DIR.exists():
        return []
    out = []
    for p in sorted(ANN_DIR.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out.append((d.get("userid", p.stem), len(d.get("judgments", {}))))
        except Exception:
            continue
    return out


# ──────────────────────────────────────────────────────────────────────────
# Guard: sample must exist
# ──────────────────────────────────────────────────────────────────────────

if not ITEMS_CSV.exists():
    st.error("Annotation sample not found.")
    st.code("python scripts/eval_concept_precision.py sample --n 200 --seed 13")
    st.stop()

items = load_items()
N = len(items)


# ──────────────────────────────────────────────────────────────────────────
# Login screen
# ──────────────────────────────────────────────────────────────────────────

if "userid" not in st.session_state:
    st.title("✍️ Concept-precision annotation")
    st.markdown(
        f"You will judge **{N} items**. Each shows one paragraph from the "
        "works of Swami Vivekananda and **one concept** the extraction "
        "pipeline attached to it.\n\n"
        "Mark **Correct** if the concept genuinely describes what the "
        "paragraph is about (at the stated relation — *discusses*, "
        "*exemplifies*, or *contrasts*), and **Incorrect** otherwise.\n\n"
        "Enter a user ID to begin. Use the **same ID** later to resume "
        "unfinished work."
    )
    uid = st.text_input(
        "Your user ID",
        placeholder="e.g. annotator_a",
        help="2–32 characters: lowercase letters, digits, - or _",
    ).strip().lower()

    users = known_users()
    if users:
        st.caption("Existing annotators: " + "  ·  ".join(
            f"`{u}` ({n}/{N})" for u, n in users))

    if st.button("Start / Resume", type="primary", use_container_width=True):
        if not _USERID_RE.match(uid):
            st.warning("Invalid ID — use 2–32 lowercase letters, digits, - or _.")
        else:
            st.session_state["userid"] = uid
            data = load_user(uid)
            save_user(data)   # registers the user on first login
            st.rerun()
    st.stop()


# ──────────────────────────────────────────────────────────────────────────
# Annotation screen
# ──────────────────────────────────────────────────────────────────────────

userid = st.session_state["userid"]
data = load_user(userid)
judgments: dict = data["judgments"]
done = len(judgments)


def first_unanswered() -> int | None:
    for it in items:
        if it["item_id"] not in judgments:
            return int(it["item_id"])
    return None


if "current" not in st.session_state:
    st.session_state["current"] = first_unanswered() or 1

with st.sidebar:
    st.markdown(f"**Annotator:** `{userid}`")
    st.progress(done / N, text=f"{done} / {N} judged")
    if st.button("Log out", use_container_width=True):
        for k in ("userid", "current"):
            st.session_state.pop(k, None)
        st.rerun()
    st.markdown("---")
    goto = st.number_input("Go to item", min_value=1, max_value=N,
                           value=int(st.session_state["current"]))
    if goto != st.session_state["current"]:
        st.session_state["current"] = int(goto)
        st.rerun()
    st.caption(
        "**1 / Correct** — the concept is a correct reading of the "
        "paragraph at the stated relation.\n\n"
        "**0 / Incorrect** — the concept is wrong, too tangential, or the "
        "relation is wrong."
    )

if done == N:
    st.success(f"🎉 All {N} items judged. Thank you!")
    st.markdown(
        "When the **second** annotator has also finished, compute the "
        "results with:\n"
        "```bash\npython scripts/eval_concept_precision.py compute\n```"
    )
    st.markdown("You can still revisit any item from the sidebar to revise "
                "a judgment.")

cur = int(st.session_state["current"])
item = items[cur - 1]
key = item["item_id"]
prev_j = judgments.get(key)

flag = "🇬🇧 English" if item["lang"] == "en" else "🇧🇩 বাংলা"
st.markdown(f"#### Item {cur} of {N} &nbsp;·&nbsp; {flag}")

st.markdown(
    f"**Concept:** &nbsp; `{item['concept']}` &nbsp;&nbsp; "
    f"**Relation:** _{item['relation']}_",
)
with st.container(border=True):
    st.write(item["paragraph_text"])

if prev_j is not None:
    st.info(f"Your current judgment: "
            f"{'✅ Correct (1)' if prev_j == 1 else '❌ Incorrect (0)'} "
            f"— click a button to change it.")

col1, col2 = st.columns(2)


def _record(value: int) -> None:
    judgments[key] = value
    save_user(data)
    nxt = first_unanswered()
    st.session_state["current"] = nxt if nxt is not None else cur
    st.rerun()


with col1:
    if st.button("✅ Correct (1)", type="primary", use_container_width=True):
        _record(1)
with col2:
    if st.button("❌ Incorrect (0)", use_container_width=True):
        _record(0)

nav1, nav2, nav3 = st.columns(3)
with nav1:
    if st.button("← Previous", disabled=cur <= 1, use_container_width=True):
        st.session_state["current"] = cur - 1
        st.rerun()
with nav2:
    if st.button("Skip →", disabled=cur >= N, use_container_width=True):
        st.session_state["current"] = cur + 1
        st.rerun()
with nav3:
    if st.button("Next unanswered ↷", use_container_width=True,
                 disabled=first_unanswered() is None):
        st.session_state["current"] = first_unanswered()
        st.rerun()
