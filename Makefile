.PHONY: data-phone train-phone api

data-phone:
\tpython build_dataset.py --sounds sounds/phone/phone_mic --out data/phone_mic

train-phone:
\tpython train_baseline.py --data data/phone_mic --out models/phone_mic

api:
\tuvicorn webapi.serve:app --reload
