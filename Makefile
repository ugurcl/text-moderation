.PHONY: install train test run api streamlit docker lint format clean

install:
	pip install -r requirements.txt
	pre-commit install

train:
	python scripts/train.py

test:
	pytest tests/ -v

run:
	python run.py

api:
	python api.py

streamlit:
	streamlit run app.py

docker:
	docker-compose up --build

lint:
	flake8 src/ scripts/ tests/ api.py app.py run.py
	isort --check-only src/ scripts/ tests/ api.py app.py run.py

format:
	black src/ scripts/ tests/ api.py app.py run.py
	isort src/ scripts/ tests/ api.py app.py run.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f data/predictions.db
