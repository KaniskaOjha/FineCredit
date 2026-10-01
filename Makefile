.PHONY: setup data train evaluate audit explain api dashboard
setup:
	python -m pip install -r requirements.txt
data:
	python -m src.data_generation
train:
	python -m src.train
evaluate:
	python -m src.evaluate
audit:
	python -m src.fairness_audit
explain:
	python -m src.explain
api:
	uvicorn src.api:app --reload
dashboard:
	streamlit run app/app.py
