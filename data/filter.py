import json

def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data

def save_to_jsonl(data, save_path):
    with open(save_path, 'w') as fout:
        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

data_path = '/home/yjiang/myWork/GainRAG/data/train_data_origin/hotpot_train_1.jsonl'
data = load_data(data_path)
item = data[0]
print(item['question'])
print(item['answers'])

new_data = []
for item in data:
    new_item={
        'Q':item['question'],
        'answers':item['answers']
    }
    new_data.append(new_item)
save_path = '/home/yjiang/myWork/Simple-AgenticRAG-RL/data/hotpot_5k.jsonl'
save_to_jsonl(new_data, save_path)
