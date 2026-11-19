.PHONY: install install-sbert train test gate api dashboard slack simulate docker docker-light

install:
	pip install -r requirements.txt && pip install -e .

install-sbert:
	pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-sbert.txt

train:
	sh scripts/train_all.sh

test:
	pytest

gate:
	python eval/check_gate.py --model-dir artifacts/robust

api:
	uvicorn comms_triage.api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

slack:
	python -m comms_triage.integrations.slack_bot

simulate:
	python scripts/simulate.py --n 40

docker:
	docker compose up --build

docker-light:
	INSTALL_SBERT=false EMBEDDING_BACKEND=tfidf docker compose up --build
