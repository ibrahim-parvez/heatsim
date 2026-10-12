PY ?= python

.PHONY: setup test bench bench-tiles scaling plots demo serve up down load clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

test:
	$(PY) -m pytest -q

bench:
	$(PY) benchmarks/bench_single.py

bench-tiles:
	$(PY) benchmarks/bench_tiles.py

scaling:
	./scripts/run_scaling.sh

plots:
	$(PY) benchmarks/plot_results.py

demo:
	$(PY) -m heatsim run --n 512 --steps 20000 --init hotspot --gif docs/hotspot.gif --frames 50
	$(PY) -m heatsim run --n 512 --steps 40000 --init edge --png docs/edge.png

# native API on the mac (can use the metal backends). port 8001 so it doesn't clash with the docker one
serve:
	$(PY) -m uvicorn service.app:app --port 8001

up:
	docker compose up --build -d

down:
	docker compose down

load:
	$(PY) scripts/load_test.py --jobs 40 --delay 1.5

clean:
	rm -rf results/jobs results/plots .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
