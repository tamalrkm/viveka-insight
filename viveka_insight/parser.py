"""HTML parsers for the English and Bengali Vivekananda complete works.

This is the existing parser (it already handles both source files cleanly),
extended with:
  * `volume_num`: an arabic-numeral normalization so "Volume V" and "খণ্ড ৫"
    can be related across languages at the volume level.
  * `iter_chapters`: groups consecutive paragraphs from the same chapter so
    chapter-level summaries / embeddings can be produced without a second pass.

Walks the DOM directly (no regex over HTML) so structural fragility is low —
if the source files change, you'll get a clear `KeyError` from BeautifulSoup,
not silently missing data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Iterator, List, Optional

from bs4 import BeautifulSoup, Tag


# ──────────────────────────────────────────────────────────────────────────
# Records
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Paragraph:
    lang: str
    volume: str                      # raw, as printed: "Volume I" / "খণ্ড ১"
    volume_num: int                  # arabic int; cross-lingual key
    section: Optional[str]           # English section header, if any
    chapter: str
    chapter_id: str                  # html id, e.g. "ch_0"
    paragraph_idx: int               # 0-based index within chapter
    text: str
    char_offset: int                 # byte offset of the source <p>
    para_id_html: str = ""           # html id of the source <p>, e.g. "p-42" (if any)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChapterGroup:
    """All paragraphs of a single chapter, for chapter-level processing."""
    lang: str
    volume: str
    volume_num: int
    section: Optional[str]
    chapter: str
    chapter_id: str
    paragraphs: List[Paragraph] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.paragraphs)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

_WS_RE = re.compile(r"\s+")

def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def _looks_meaningful(text: str, min_chars: int = 20) -> bool:
    if not text or len(text) < min_chars:
        return False
    if text.strip() in {"←", "→", "↑", "↓"}:
        return False
    return True


# Map English roman numerals to int. Limited to what actually appears in the
# corpus (volumes I–IX); fail loudly on anything unexpected.
_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
          "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}

def _roman_to_int(s: str) -> int:
    s = s.strip().upper()
    if s in _ROMAN:
        return _ROMAN[s]
    # numeric (defensive)
    try:
        return int(s)
    except ValueError:
        return 0


# Bengali digits (০-৯) and Arabic-Indic alternates: convert to int.
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def _bengali_to_int(s: str) -> int:
    s = s.translate(_BN_DIGITS).strip()
    digits = re.sub(r"\D", "", s)
    return int(digits) if digits else 0


# ──────────────────────────────────────────────────────────────────────────
# English parser
# ──────────────────────────────────────────────────────────────────────────

# "Volume I", "Volume II", ...
_EN_VOL_RE = re.compile(r"Volume\s+([IVX]+)\b", re.IGNORECASE)

def parse_english(html_path: str) -> Iterator[Paragraph]:
    """Walk vivekananda_complete_works.html, yield Paragraph for each <p>.

    The source file's structure is a flat sibling sequence under <body>:
      <h2 class="volume-title">Volume I</h2>
      <div class="section-head">…</div>
      <div class="subsection-head">…</div>      (optional, qualifies section)
      <div class="chapter">
        <h3 class="chapter-title">…</h3>
        <p class="chapter-breadcrumb">…</p>     (skip)
        <div class="chapter-content"> … <p>…</p> … </div>
        <a class="back-top">…</a>               (skip)
      </div>
      …
      <h2 class="volume-title">Volume II</h2>
      …
    There is no enclosing volume <div>, so we walk transition nodes in
    document order, tracking current volume / section / subsection. Chapter
    HTML ids are absent in this file, so we synthesize "ch_<vol>_<ord>".
    """
    with open(html_path, "r", encoding="utf-8") as f:
        raw = f.read()

    soup = BeautifulSoup(raw, "lxml")
    cursor = 0  # advancing offset cursor for O(n) char_offset lookup

    volume = "Unknown Volume"
    volume_num = 0
    current_section: Optional[str] = None
    section_root: Optional[str] = None     # last section-head; subsection appends to it
    chapter_ord = 0                         # chapter ordinal within current volume

    nodes = soup.find_all(
        ["h2", "div"],
        class_=["volume-title", "section-head", "subsection-head", "chapter"],
    )

    for node in nodes:
        classes = node.get("class") or []

        if node.name == "h2" and "volume-title" in classes:
            volume = _clean(node.get_text()) or "Unknown Volume"
            m = _EN_VOL_RE.search(volume)
            volume_num = _roman_to_int(m.group(1)) if m else 0
            current_section = None
            section_root = None
            chapter_ord = 0
            continue

        if "section-head" in classes:
            section_root = _clean(node.get_text()) or None
            current_section = section_root
            continue

        if "subsection-head" in classes:
            sub = _clean(node.get_text())
            if sub:
                current_section = f"{section_root} / {sub}" if section_root else sub
            continue

        if "chapter" not in classes:
            continue

        title_tag = node.find("h3", class_="chapter-title")
        chapter = _clean(title_tag.get_text()) if title_tag else "Untitled"
        chapter_id = node.get("id") or f"ch_{volume_num}_{chapter_ord}"
        chapter_ord += 1

        content_div = node.find("div", class_="chapter-content")
        if not content_div:
            continue

        p_idx = 0
        for p in content_div.find_all("p"):
            p_classes = p.get("class") or []
            if "chapter-breadcrumb" in p_classes or "nav" in p_classes:
                continue
            text = _clean(p.get_text(separator=" "))
            if not _looks_meaningful(text):
                continue
            snippet = str(p)[:80]
            offset = raw.find(snippet, cursor)
            if offset >= 0:
                cursor = offset
            yield Paragraph(
                lang="en",
                volume=volume,
                volume_num=volume_num,
                section=current_section,
                chapter=chapter,
                chapter_id=chapter_id,
                paragraph_idx=p_idx,
                text=text,
                char_offset=offset if offset >= 0 else cursor,
                para_id_html=p.get("id") or "",
            )
            p_idx += 1


# ──────────────────────────────────────────────────────────────────────────
# Bengali parser
# ──────────────────────────────────────────────────────────────────────────

# "স্বামী বিবেকানন্দ সমগ্র খন্ড ১", with তslightly variant spelling of "khanda"
_BN_VOL_H3_RE = re.compile(
    r"স্বামী\s+বিবেকানন্দ\s+সমগ্র\s+(খন্ড|খণ্ড)\s+([\u09E6-\u09EF০-৯0-9]+)"
)


def _bengali_volume_boundaries(soup: BeautifulSoup, raw: str):
    """List of (offset_in_raw, label, volume_num) sorted by offset."""
    boundaries = []
    for h3 in soup.find_all("h3"):
        txt = _clean(h3.get_text())
        m = _BN_VOL_H3_RE.search(txt)
        if not m:
            continue
        offset = raw.find(str(h3))
        if offset < 0:
            offset = raw.find(f">{txt}<")
        num_raw = m.group(2)
        vol_num = _bengali_to_int(num_raw)
        label = f"খণ্ড {num_raw}"
        boundaries.append((offset if offset >= 0 else 0, label, vol_num))
    boundaries.sort()
    return boundaries


def _volume_for_offset(offset: int, boundaries):
    current_label, current_num = "খণ্ড ১", 1
    for start, label, num in boundaries:
        if start <= offset:
            current_label, current_num = label, num
        else:
            break
    return current_label, current_num


def parse_bengali(html_path: str) -> Iterator[Paragraph]:
    """Walk all.html, yield Paragraph for each meaningful <p>."""
    with open(html_path, "r", encoding="utf-8") as f:
        raw = f.read()

    soup = BeautifulSoup(raw, "lxml")
    boundaries = _bengali_volume_boundaries(soup, raw)

    cursor = 0
    chapter_cursor = 0

    # Chapter container: the original export used <div class="bb-item">; a later
    # re-export uses <section class="content-block"> (same id="content-N", same
    # inner <div class="scroller">). Accept either so both HTML vintages parse.
    chapters = (soup.find_all("div", class_="bb-item")
                or soup.find_all("section", class_="content-block"))
    for bb in chapters:
        chapter_id = bb.get("id") or ""

        scroller = bb.find("div", class_="scroller") or bb
        h2 = scroller.find("h2")
        chapter_title = _clean(h2.get_text()) if h2 else ""
        if not chapter_title:
            chapter_title = "(শিরোনামহীন)"

        bb_offset = raw.find(f'id="{chapter_id}"', chapter_cursor)
        if bb_offset >= 0:
            chapter_cursor = bb_offset
            cursor = bb_offset
        else:
            bb_offset = chapter_cursor
        volume_label, volume_num = _volume_for_offset(bb_offset, boundaries)

        p_idx = 0
        for p in scroller.find_all("p"):
            text = _clean(p.get_text(separator=" "))
            if not _looks_meaningful(text):
                continue
            snippet = str(p)[:80]
            offset = raw.find(snippet, cursor)
            if offset >= 0:
                cursor = offset
            yield Paragraph(
                lang="bn",
                volume=volume_label,
                volume_num=volume_num,
                section=None,
                chapter=chapter_title,
                chapter_id=chapter_id,
                paragraph_idx=p_idx,
                text=text,
                char_offset=offset if offset >= 0 else cursor,
                para_id_html=p.get("id") or "",
            )
            p_idx += 1


# ──────────────────────────────────────────────────────────────────────────
# Chapter-level grouping
# ──────────────────────────────────────────────────────────────────────────

def iter_chapters(paragraphs: Iterator[Paragraph]) -> Iterator[ChapterGroup]:
    """Group an in-order paragraph stream by (chapter_id, volume).

    Two chapters with the same html id from different volumes do exist (ids
    reset between volumes for one of the source files), so we key on the pair.
    """
    current: Optional[ChapterGroup] = None
    for p in paragraphs:
        key = (p.volume, p.chapter_id, p.chapter)
        if current is None or (current.volume, current.chapter_id, current.chapter) != key:
            if current is not None:
                yield current
            current = ChapterGroup(
                lang=p.lang,
                volume=p.volume,
                volume_num=p.volume_num,
                section=p.section,
                chapter=p.chapter,
                chapter_id=p.chapter_id,
            )
        current.paragraphs.append(p)
    if current is not None:
        yield current


# ──────────────────────────────────────────────────────────────────────────
# Sentence splitting
# ──────────────────────────────────────────────────────────────────────────

# Bengali terminators: ।  (U+0964 daari), plus ?  !
_BN_SENT_RE = re.compile(r"(?<=[।?!])\s+")

def split_sentences_bn(text: str) -> List[str]:
    parts = _BN_SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def split_sentences_en(text: str) -> List[str]:
    """Use NLTK punkt if available; fall back to a regex."""
    try:
        import nltk
        try:
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
            return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except Exception:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", text)
        return [p.strip() for p in parts if p.strip()]


def split_sentences(text: str, lang: str) -> List[str]:
    if lang == "bn":
        return split_sentences_bn(text)
    return split_sentences_en(text)
