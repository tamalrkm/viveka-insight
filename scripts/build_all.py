"""Run the whole pipeline end-to-end:  parse → embed → extract → link → check → (gnn).

Each stage is a separate Python script invoked as a subprocess. This isolates
GPU memory between stages — the embedder is fully unloaded before vLLM tries
to grab the GPU for the LLM, etc. — and gives clean, separable logs.

All stages are idempotent and resumable. If you SIGKILL the orchestrator,
re-running it will skip everything already complete and pick up where the
last stage was interrupted.

Pass --skip-llm to skip stage 3 (concept extraction) when you only want a
quick sentence-level vector index without the full graph. Pass --gnn to
include the optional GNN training step.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))

from viveka_insight import db as dbmod  # noqa: E402
from viveka_insight.config import CFG  # noqa: E402


STAGES = [
    # 00: dump existing per-paragraph extractions into the warm-start cache
    # before any wipe. Cheap; no-op on first build (empty DB).
    ("00_snapshot",         ["python", str(SCRIPTS / "snapshot_concepts.py")]),
    ("01_parse",            ["python", str(SCRIPTS / "01_parse.py")]),
    ("02_embed",            ["python", str(SCRIPTS / "02_embed.py")]),
    # 02b: re-attach cached extractions to fresh paragraph rows by text match.
    # No LLM; matters only on incremental rebuilds.
    ("02b_restore",         ["python", str(SCRIPTS / "restore_concepts.py")]),
    ("03_extract_concepts", ["python", str(SCRIPTS / "03_extract_concepts.py")]),
    ("04_link_concepts",    ["python", str(SCRIPTS / "04_link_concepts.py")]),
    ("05_build_graph",      ["python", str(SCRIPTS / "05_build_graph.py")]),
]
GNN_STAGE = ("06_train_gnn", ["python", str(SCRIPTS / "06_train_gnn.py")])


def _run(name: str, cmd: list, env: dict) -> int:
    print("\n" + "─" * 70)
    print(f"  STAGE: {name}")
    print("─" * 70)
    t0 = time.time()
    rc = subprocess.call(cmd, env=env, cwd=str(ROOT))
    dt = time.time() - t0
    if rc != 0:
        print(f"\n  ✗ stage {name} failed with exit code {rc} after {dt:.0f}s")
    else:
        print(f"\n  ✓ stage {name} completed in {dt:.0f}s")
    return rc


def main():
    ap = argparse.ArgumentParser(description="Run the full viveka-insight build")
    ap.add_argument("--skip-llm", action="store_true",
                    help="skip concept extraction (much faster, vector-only index)")
    ap.add_argument("--gnn", action="store_true",
                    help="include the optional GNN training stage")
    ap.add_argument("--from-stage", default=None,
                    help="skip stages until this name (for debugging re-runs)")
    ap.add_argument("--sample", action="store_true",
                    help="build the tiny synthetic sample corpus into "
                         "index_data_sample/ using the stub LLM backend "
                         "(no GPU, no corpus, no LLM download)")
    args = ap.parse_args()

    env = os.environ.copy()
    if args.gnn:
        env["VIVEKA_GNN_ENABLED"] = "1"

    if args.sample:
        sample_dir = ROOT / "sample_data"
        en, bn = sample_dir / "sample_en.html", sample_dir / "sample_bn.html"
        missing = [p for p in (en, bn) if not p.exists()]
        if missing:
            sys.exit("  ✗ sample corpus missing: "
                     + ", ".join(str(p) for p in missing))
        # Redirect *everything* so a sample build can never touch a real index.
        env["VIVEKA_EN_HTML"] = str(en)
        env["VIVEKA_BN_HTML"] = str(bn)
        env["VIVEKA_DATA_DIR"] = str(sample_dir)
        env["VIVEKA_INDEX_DIR"] = str(ROOT / "index_data_sample")
        env["VIVEKA_LLM_BACKEND"] = "stub"
        env.setdefault("VIVEKA_DEVICE", "cpu")
        print("═" * 70)
        print("  SAMPLE BUILD")
        print("═" * 70)
        print("  corpus     : sample_data/ (40 synthetic paragraphs, EN + BN)")
        print("  index dir  : index_data_sample/  (your real index is untouched)")
        print("  LLM        : stub — a keyword matcher, NOT a language model")
        print(f"  device     : {env['VIVEKA_DEVICE']}")
        print("\n  This demonstrates that the pipeline runs. The resulting")
        print("  concept graph is illustrative only and must not be searched,")
        print("  evaluated or published as if it were real output.\n")

    started = args.from_stage is None
    stages = list(STAGES)
    if args.gnn:
        stages.append(GNN_STAGE)

    overall_t0 = time.time()
    for name, cmd in stages:
        if not started:
            if args.from_stage and args.from_stage in name:
                started = True
            else:
                print(f"  ⏭  skipping {name} (--from-stage={args.from_stage})")
                continue
        if args.skip_llm and name in ("00_snapshot", "02b_restore",
                                       "03_extract_concepts", "04_link_concepts"):
            print(f"  ⏭  skipping {name} (--skip-llm)")
            continue
        rc = _run(name, cmd, env)
        if rc != 0:
            print("\n  Build halted. Fix the error above, then rerun build_all.py "
                  "(completed stages will be skipped).")
            sys.exit(rc)

    print("\n" + "═" * 70)
    print(f"  ✓ Pipeline complete in {(time.time()-overall_t0)/60:.1f} min")
    print("═" * 70)
    if args.sample:
        index_dir = ROOT / "index_data_sample"
        print(f"  Index dir:   {index_dir}")
        print(f"  DB:          {index_dir / 'meta.sqlite'}")
        print("\n  Inspect what was built:")
        print(f"     sqlite3 {index_dir / 'meta.sqlite'} \\")
        print("       'SELECT co.canonical_label, COUNT(*) FROM para_concept pc")
        print("        JOIN concepts co ON co.id=pc.concept_id")
        print("        GROUP BY 1 ORDER BY 2 DESC;'")
        print("\n  Browse it in the webapp:")
        print(f"     VIVEKA_INDEX_DIR={index_dir} VIVEKA_DEVICE=cpu \\")
        print("       streamlit run webapp/app.py --server.port 8501")
        print("\n  Reminder: stub-LLM output. Illustrative only.")
    else:
        print(f"  Index dir:   {CFG.paths.index_dir}")
        print(f"  DB:          {CFG.paths.db}")
        print(f"\n  Launch the webapp:")
        print(f"     streamlit run webapp/app.py --server.address 0.0.0.0 --server.port 8501")


if __name__ == "__main__":
    main()
