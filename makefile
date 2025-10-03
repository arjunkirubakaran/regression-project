# Makefile for regression project

PYTHON=python3
SCRIPT=training.py
DATA=ENB2012_data.xlsx
OUT=outputs

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) $(SCRIPT) --data $(DATA) --out $(OUT) --seed 42 --alphas "0,0.1,1,10,100"

clean:
	rm -rf $(OUT)/*.csv $(OUT)/*.json
