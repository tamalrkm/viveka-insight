"""Strip dead/noisy elements from data/all.html and merge to one document.

The Bengali source is literally 10 BookBlock-templated single-volume HTML
documents concatenated end-to-end (`</body></html><!DOCTYPE…<html><head>…
</head><body>` boilerplate between each). On top of that there are:

  * 10× <audio controls autoplay> pointing at /static/cwsv/audio/khandana.mp3
    (the mp3 itself returns 404 on the host; only the broken player bar shows)
  * 76× <img> tags whose sources all 404 on the host (one is even a Windows
    local path leak: `C:\\Users\\hirak\\…`)
  * a typo'd favicon path ("/staic/cwsv/favicon.ico") repeated per volume
  * 1,917 HTML comments (~67 KB of template residue)

This script flattens all 10 volume documents into a single well-formed HTML
page (one <head>, one <body>) and strips the dead media/comments. It does not
touch a single character of <p> text content — the parser only walks <p>
nodes — so a re-parse + warm-start restore should match every Bengali
paragraph by text and trigger zero new LLM calls.

Behaviour:
  - Saves the original to data/all.html.bak on first run; aborts (safely) if
    the backup already exists *unless* you pass --force-backup-overwrite.
  - Writes the cleaned content back to data/all.html.
  - Prints a before/after stats summary.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parent.parent / "data" / "all.html"


_AUDIO_RE   = re.compile(r"<audio\b[^>]*>.*?</audio>", re.IGNORECASE | re.DOTALL)
_IMG_RE     = re.compile(r"<img\b[^>]*/?>", re.IGNORECASE)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_FAVICON_RE = re.compile(
    r'<link\b[^>]*rel="shortcut icon"[^>]*>', re.IGNORECASE,
)
# Source typo right before Volume 6's heading: a stray `<b>` written where
# `</b>` was meant, leaving a bold context permanently open. Browser then
# renders the rest of the corpus bold. Fix every `<b>(whitespace?)</p>` —
# there is exactly one such occurrence and it would never be intentional.
_BOLD_TYPO_RE = re.compile(r"<b>(\s*)</p>", re.IGNORECASE)

# Match the inter-volume document fence: closing one volume's <body>/<html>,
# then a fresh <!DOCTYPE>/<html>/<head>…</head>/<body>. The inner <head> can be
# 100s of KB (links + scripts), so DOTALL is needed.
_VOL_BOUNDARY_RE = re.compile(
    r"</body>\s*</html>"                # close prior volume
    r"\s*<!DOCTYPE[^>]*>"               # next volume's doctype
    r"\s*<html\b[^>]*>"                 # <html ...>
    r"\s*<head\b[^>]*>.*?</head>"       # discard the next head wholesale
    r"\s*<body\b[^>]*>",                # open next volume's body
    re.IGNORECASE | re.DOTALL,
)


def clean(raw: str) -> tuple[str, dict]:
    stats = {}
    out = raw

    # Merge volumes first — this also takes care of duplicate <head> blocks
    # 2..10, duplicate doctypes, and the redundant <html>/<body> fences.
    out, n = _VOL_BOUNDARY_RE.subn("\n", out); stats["volume_boundaries_merged"] = n

    out, n = _AUDIO_RE.subn("", out);   stats["audio_blocks"] = n
    out, n = _IMG_RE.subn("", out);     stats["img_tags"]     = n
    out, n = _COMMENT_RE.subn("", out); stats["comments"]     = n
    out, n = _FAVICON_RE.subn("", out); stats["favicon_links"] = n
    out, n = _BOLD_TYPO_RE.subn(r"</b>\1</p>", out); stats["bold_typo_fixes"] = n

    return out, stats


def main():
    ap = argparse.ArgumentParser(description="Strip dead media + comments from data/all.html")
    ap.add_argument("--path", default=str(SRC),
                    help="HTML file to clean (default: data/all.html)")
    ap.add_argument("--force-backup-overwrite", action="store_true",
                    help="overwrite data/all.html.bak if it exists")
    ap.add_argument("--dry-run", action="store_true",
                    help="print stats but don't write the file")
    args = ap.parse_args()

    src = Path(args.path)
    bak = src.with_suffix(src.suffix + ".bak")

    raw = src.read_text(encoding="utf-8")
    before = len(raw)
    cleaned, stats = clean(raw)
    after = len(cleaned)

    print(f"file: {src}")
    print(f"  before: {before:>10,} chars")
    print(f"  after:  {after:>10,} chars")
    print(f"  saved:  {before - after:>10,} chars ({100 * (before - after) / before:.1f}%)")
    print(f"  stripped:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    if args.dry_run:
        print("\n(dry run — file not written)")
        return

    if bak.exists() and not args.force_backup_overwrite:
        print(f"\nbackup already exists: {bak}")
        print("(refusing to overwrite. Pass --force-backup-overwrite to replace it.)")
    else:
        shutil.copy2(src, bak)
        print(f"\nbackup written: {bak}")

    src.write_text(cleaned, encoding="utf-8")
    print(f"cleaned file written: {src}")


if __name__ == "__main__":
    main()
