# viveka-insight — common workflows
.PHONY: help install build build-fast build-gnn parse embed concepts link graph gnn webapp test clean clean-index

PY ?= python

help:
	@echo "make install       — install Python deps (assumes torch already installed)"
	@echo "make build         — full pipeline: parse + embed + concepts + link + check"
	@echo "make build-fast    — pipeline without LLM concept extraction (vector-only)"
	@echo "make build-gnn     — full pipeline + optional GNN refinement step"
	@echo ""
	@echo "make parse         — stage 1 only"
	@echo "make embed         — stage 2 only"
	@echo "make concepts      — stage 3 only (LLM)"
	@echo "make link          — stage 4 only"
	@echo "make graph         — stage 5 only"
	@echo "make gnn           — stage 6 only (optional)"
	@echo ""
	@echo "make webapp        — launch the Streamlit UI on 0.0.0.0:8501"
	@echo "make test          — run the smoke test suite"
	@echo "make clean-index   — wipe index_data/ (forces full rebuild)"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -c "import nltk; nltk.download('punkt_tab')"

build:
	$(PY) scripts/build_all.py

build-fast:
	$(PY) scripts/build_all.py --skip-llm

build-gnn:
	$(PY) scripts/build_all.py --gnn

parse:
	$(PY) scripts/01_parse.py

embed:
	$(PY) scripts/02_embed.py

concepts:
	$(PY) scripts/03_extract_concepts.py

link:
	$(PY) scripts/04_link_concepts.py

graph:
	$(PY) scripts/05_build_graph.py

gnn:
	VIVEKA_GNN_ENABLED=1 $(PY) scripts/06_train_gnn.py --enable

webapp:
	streamlit run webapp/app.py --server.address 0.0.0.0 --server.port 8501

test:
	$(PY) -m pytest tests/ -x -q

clean-index:
	rm -rf index_data/
	@echo "Wiped index_data/. Re-run 'make build' for a fresh index."
