import random
import os, random, re, requests, time, json
import torch
from torch.nn.utils.rnn import pad_sequence
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from agent import gen_sample


model_path = '/home/share/models/Qwen2.5-3B-Instruct'
gen = LLM(model= model_path, gpu_memory_utilization=0.85, max_model_len=1024)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# QAs = [ {'Q':'what is python', 'A':'xx'}, {'Q':'what is English', 'A':'xx'},]
def load_data(data_path='/home/yjiang/myWork/Simple-AgenticRAG-RL/data/hotpot_5k.jsonl'):
    with open(data_path, 'r') as f:
        QAs = [json.loads(line) for line in f]
    return QAs
QAs = random.sample(load_data(),3)
_, answers, ans_token_ids ,ans_masks = gen_sample(gen,tokenizer, QAs, 8)


tensor_ans_ids = [torch.tensor(ans_ids) for ans_ids in ans_token_ids]
output_ids = pad_sequence(tensor_ans_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
tensor_ans_masks = [torch.tensor(masks) for masks in ans_masks]
mask_ids = pad_sequence(tensor_ans_masks, batch_first=True, padding_value=0)

print(output_ids.shape)
print(mask_ids.shape)

for i in range(len(ans_masks)):
    print(answers[i])
    print('-'*100)
    for idx, mask_val in enumerate(ans_masks[i]):
        if mask_val == 0:
            token_id = ans_token_ids[i][idx]
            token_str = tokenizer.decode([token_id])
            print(f"{token_str}", end='')
    print('')
    print('x'*100)

    