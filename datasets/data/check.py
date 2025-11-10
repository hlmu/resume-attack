import json
import os
# traverse all json files in train_data and print the data

for file in os.listdir('train_data'):
    with open(f'train_data/{file}', 'r') as f:
        data = json.load(f)
    for item in data:
        assert len(item) == 4
        assert item[0]['role'] == "system"
        assert item[1]['role'] == "user"
        assert item[2]['role'] == "data"

for file in os.listdir('eval_data'):
    with open(f'eval_data/{file}', 'r') as f:
        data = json.load(f)
    for item in data:
        assert len(item) == 3
