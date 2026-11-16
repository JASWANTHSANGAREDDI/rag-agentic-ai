.PHONY: install ingest run dev

install:
	pip install -r requirements.txt

ingest:
	python src/ingestion.py

run:
	uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4

dev:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload
