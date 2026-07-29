"""Concept-extraction precision: annotation kit (paper Evaluation).

This study needs HUMAN judgments — do not fill the sheets with a model.

Step 1 — create the sample (run once):
    python scripts/eval_concept_precision.py sample --n 200 --seed 13
  Writes docs/paper/eval/concept_sample_annotator_A.csv and _B.csv
  (identical items, independent copies). Each row shows a paragraph and ONE
  concept attached to it, with its relation. The annotator fills the
  `judgment` column with 1 (the concept is a correct reading of the
  paragraph, at the stated relation) or 0 (incorrect / not supported).

Step 2 — collect judgments, either way:
  * Web UI (recommended): `bash run_annotate.sh`, each annotator logs in
    with their own user ID at http://<host>:8502 and can resume any time.
    Judgments land in docs/paper/eval/annotations/<userid>.json.
  * Or fill the `judgment` column (1/0) in the two CSV sheets by hand.

Step 3 — when two annotators have finished:
    python scripts/eval_concept_precision.py compute [--users alice bob]
  UI annotations take precedence when present (with --users, or
  automatically when exactly two users have judged every item); otherwise
  the CSV sheets are read. Prints precision (consensus = both say correct;
  also each annotator's precision) and Cohen's kappa, and writes
  eval_concept_precision.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from viveka_insight.config import CFG

EVAL_DIR = ROOT / "docs" / "paper" / "eval"
SHEETS = [EVAL_DIR / f"concept_sample_annotator_{a}.csv" for a in ("A", "B")]
OUT_JSON = EVAL_DIR / "eval_concept_precision.json"

FIELDS = ["item_id", "lang", "concept", "relation", "weight",
          "paragraph_text", "judgment"]


def sample(n: int, seed: int) -> None:
    conn = sqlite3.connect(CFG.paths.db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT pc.paragraph_id, pc.concept_id, pc.relation, pc.weight, "
        "       co.canonical_label, p.text, b.lang "
        "FROM para_concept pc "
        "JOIN concepts co ON co.id = pc.concept_id "
        "JOIN paragraphs p ON p.id = pc.paragraph_id "
        "JOIN chapters c ON c.id = p.chapter_id "
        "JOIN volumes v ON v.id = c.volume_id "
        "JOIN books b ON b.id = v.book_id"
    ).fetchall()
    random.Random(seed).shuffle(rows)
    picked = rows[:n]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for sheet in SHEETS:
        with sheet.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for i, r in enumerate(picked, 1):
                w.writerow({
                    "item_id": i,
                    "lang": r["lang"],
                    "concept": r["canonical_label"],
                    "relation": r["relation"],
                    "weight": r["weight"],
                    "paragraph_text": r["text"][:1200],
                    "judgment": "",
                })
        print(f"wrote {sheet}  ({len(picked)} items)")
    print("\nHand one sheet to each annotator. Judgment: 1 = concept correctly "
          "describes the paragraph (at the stated relation), 0 = incorrect.")


ANN_DIR = EVAL_DIR / "annotations"


def _read_judgments(path: Path) -> dict:
    out = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            j = row["judgment"].strip()
            if j not in ("0", "1"):
                raise SystemExit(
                    f"{path}: item {row['item_id']} has judgment {j!r} "
                    f"(every row must be 0 or 1)")
            out[row["item_id"]] = int(j)
    return out


def _item_ids() -> set:
    with SHEETS[0].open(encoding="utf-8") as f:
        return {row["item_id"] for row in csv.DictReader(f)}


def _read_ui_user(userid: str) -> dict:
    p = ANN_DIR / f"{userid}.json"
    if not p.exists():
        raise SystemExit(f"no UI annotations for user {userid!r} ({p})")
    return {k: int(v)
            for k, v in json.loads(p.read_text())["judgments"].items()}


def _pick_judgment_sources(users) -> tuple[dict, dict, str]:
    """Return (judgments_a, judgments_b, description)."""
    ids = _item_ids()
    if users:
        if len(users) != 2:
            raise SystemExit("--users takes exactly two user IDs")
        a, b = (_read_ui_user(u) for u in users)
        for u, j in zip(users, (a, b)):
            missing = ids - j.keys()
            if missing:
                raise SystemExit(
                    f"user {u!r} has {len(missing)} unjudged items "
                    f"(e.g. item {sorted(missing, key=int)[0]})")
        return a, b, f"UI annotators: {users[0]}, {users[1]}"
    # auto-detect: exactly two UI users who judged everything
    if ANN_DIR.exists():
        complete = []
        for p in sorted(ANN_DIR.glob("*.json")):
            d = json.loads(p.read_text())
            if ids <= set(d.get("judgments", {}).keys()):
                complete.append(d["userid"])
        if len(complete) == 2:
            return (_read_ui_user(complete[0]), _read_ui_user(complete[1]),
                    f"UI annotators: {complete[0]}, {complete[1]}")
        if len(complete) > 2:
            raise SystemExit(
                f"{len(complete)} complete UI annotators found "
                f"({', '.join(complete)}); pick two with --users")
    # fall back to the CSV sheets
    a, b = _read_judgments(SHEETS[0]), _read_judgments(SHEETS[1])
    return a, b, "CSV sheets A, B"


# ──────────────────────────────────────────────────────────────────────────
# Statistics (stdlib only, to keep this script dependency-free)
# ──────────────────────────────────────────────────────────────────────────

def _wilson(k: int, n: int, z: float = 1.96) -> list:
    """Wilson score interval — well behaved for proportions near 0/1."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _kappa(xa, xb) -> float:
    n = len(xa)
    po = sum(x == y for x, y in zip(xa, xb)) / n
    pa, pb = sum(xa) / n, sum(xb) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def _gwet_ac1(xa, xb) -> float:
    """Prevalence-robust agreement; reported alongside kappa because kappa
    is depressed when the two raters have very different marginals."""
    n = len(xa)
    po = sum(x == y for x, y in zip(xa, xb)) / n
    pi = (sum(xa) / n + sum(xb) / n) / 2
    pe = 2 * pi * (1 - pi)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test against p=0.5 (McNemar)."""
    if n == 0:
        return 1.0
    probs = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[k] * (1 + 1e-9)))


def _fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test on the 2x2 table [[a,b],[c,d]]."""
    n = a + b + c + d
    r1, c1 = a + b, a + c
    def hyp(x):
        return (math.comb(r1, x) * math.comb(n - r1, c1 - x)
                / math.comb(n, c1))
    obs = hyp(a)
    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    return min(1.0, sum(hyp(x) for x in range(lo, hi + 1)
                        if hyp(x) <= obs * (1 + 1e-9)))


def _spearman(xs, ys) -> float:
    """Spearman rho with midranks for ties (weights are heavily tied)."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx)
                    * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def _read_items() -> dict:
    """item_id -> row, for the lang/relation/weight breakdowns."""
    with SHEETS[0].open(encoding="utf-8") as f:
        return {r["item_id"]: r for r in csv.DictReader(f)}


def compute(users=None) -> None:
    a, b, source = _pick_judgment_sources(users)
    print(f"judgment source: {source}")
    if a.keys() != b.keys():
        raise SystemExit("annotators do not cover the same items")
    ids = sorted(a, key=int)
    xa = [a[i] for i in ids]
    xb = [b[i] for i in ids]
    n = len(ids)
    cons = [x & y for x, y in zip(xa, xb)]
    union = [x | y for x, y in zip(xa, xb)]

    res = {
        "judgment_source": source,
        "n_items": n,
        "precision_annotator_A": sum(xa) / n,
        "precision_annotator_A_ci95": _wilson(sum(xa), n),
        "precision_annotator_B": sum(xb) / n,
        "precision_annotator_B_ci95": _wilson(sum(xb), n),
        "precision_consensus_both_correct": sum(cons) / n,
        "precision_consensus_ci95": _wilson(sum(cons), n),
        "precision_union_either_correct": sum(union) / n,
        "precision_union_ci95": _wilson(sum(union), n),
        "raw_agreement": sum(x == y for x, y in zip(xa, xb)) / n,
        "cohens_kappa": _kappa(xa, xb),
        "gwet_ac1": _gwet_ac1(xa, xb),
    }

    # Bootstrap CI for kappa (seeded: this number goes in the paper).
    rng = random.Random(13)
    boots = []
    for _ in range(10000):
        s = [rng.randrange(n) for _ in range(n)]
        boots.append(_kappa([xa[i] for i in s], [xb[i] for i in s]))
    boots.sort()
    res["cohens_kappa_ci95_bootstrap"] = [boots[int(0.025 * len(boots))],
                                          boots[int(0.975 * len(boots))]]

    # Severity asymmetry: is the disagreement systematic or noise?
    n_b = sum(1 for x, y in zip(xa, xb) if x == 0 and y == 1)
    n_c = sum(1 for x, y in zip(xa, xb) if x == 1 and y == 0)
    res["discordant_A0_B1"] = n_b
    res["discordant_A1_B0"] = n_c
    res["mcnemar_exact_p"] = _binom_two_sided(min(n_b, n_c), n_b + n_c)

    # Breakdowns by extractor weight / language / relation.
    rows = _read_items()
    wts = [float(rows[i]["weight"]) for i in ids]
    res["spearman_weight_vs_consensus"] = _spearman(wts, cons)
    res["spearman_weight_vs_A"] = _spearman(wts, xa)
    res["spearman_weight_vs_B"] = _spearman(wts, xb)

    def subset(mask):
        idx = [j for j, m in enumerate(mask) if m]
        if not idx:
            return None
        return {
            "n": len(idx),
            "precision_A": sum(xa[j] for j in idx) / len(idx),
            "precision_B": sum(xb[j] for j in idx) / len(idx),
            "precision_consensus": sum(cons[j] for j in idx) / len(idx),
        }

    hi = [w >= 0.8 for w in wts]
    res["by_weight"] = {"weight_ge_0.8": subset(hi),
                        "weight_lt_0.8": subset([not h for h in hi])}
    k_hi = sum(cons[j] for j, h in enumerate(hi) if h)
    k_lo = sum(cons[j] for j, h in enumerate(hi) if not h)
    res["by_weight"]["fisher_p_consensus"] = _fisher_exact(
        k_hi, sum(hi) - k_hi, k_lo, (n - sum(hi)) - k_lo)

    res["by_language"] = {
        L: subset([rows[i]["lang"] == L for i in ids]) for L in ("en", "bn")}
    en = [j for j, i in enumerate(ids) if rows[i]["lang"] == "en"]
    bn = [j for j, i in enumerate(ids) if rows[i]["lang"] == "bn"]
    k_en, k_bn = sum(cons[j] for j in en), sum(cons[j] for j in bn)
    res["by_language"]["fisher_p_consensus"] = _fisher_exact(
        k_en, len(en) - k_en, k_bn, len(bn) - k_bn)
    res["by_relation"] = {
        R: subset([rows[i]["relation"] == R for i in ids])
        for R in sorted({rows[i]["relation"] for i in ids})}

    OUT_JSON.write_text(json.dumps(res, indent=2))

    def show(d, indent=0):
        for k, v in d.items():
            pad = " " * indent
            if isinstance(v, dict):
                print(f"{pad}{k}:")
                show(v, indent + 2)
            elif isinstance(v, float):
                print(f"{pad}{k}: {v:.4g}")
            elif isinstance(v, list):
                print(f"{pad}{k}: [{v[0]:.3f}, {v[1]:.3f}]")
            else:
                print(f"{pad}{k}: {v}")

    show(res)
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sample")
    sp.add_argument("--n", type=int, default=200)
    sp.add_argument("--seed", type=int, default=13)
    cp = sub.add_parser("compute")
    cp.add_argument("--users", nargs=2, metavar=("USER_A", "USER_B"),
                    help="two annotation-UI user IDs to score")
    args = ap.parse_args()
    if args.cmd == "sample":
        sample(args.n, args.seed)
    else:
        compute(args.users)
