import json

base_dir = '../Datasets/naturalquestions/preprocess/'
data = json.load(open('../Datasets/naturalquestions/preprocess/FULL.json', 'r', encoding='utf-8'))

import random

random.shuffle(data)


n1 = 1000

test = data[:n1]
remain_data = data[n1:]

n = len(remain_data)
train_data = remain_data[:int(0.9 * n)]
valid_data = remain_data[int(0.9 * n):]

with open(f"{base_dir}/test.json", "w", encoding="utf-8") as f:
    json.dump(test, f, ensure_ascii=False, indent=4)

with open(f"{base_dir}/train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

with open(f"{base_dir}/valid.json", "w", encoding="utf-8") as f:
    json.dump(valid_data, f, ensure_ascii=False, indent=4)


print("SPLIT DONE!!")
