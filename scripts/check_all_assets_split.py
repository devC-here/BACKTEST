import json

reg = json.load(open('data_splits/clean_weeks_registry.json'))
first_asset = list(reg['assets'].keys())[0]
print(f"Sample asset '{first_asset}':", reg['assets'][first_asset].keys())
print(reg['assets'][first_asset])
