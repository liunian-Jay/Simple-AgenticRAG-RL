import random
import os, random, re, requests, time, json
import torch
from torch.nn.utils.rnn import pad_sequence
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from agent import gen_sample
from reward import em_max_over_ground_truths, f1_max_over_ground_truths

def load_data(data_path):
    with open(data_path, 'r') as f:
        QAs = [json.loads(line) for line in f]
    return QAs

def main(model_path, data_path):
    gen = LLM(model= model_path, gpu_memory_utilization=0.85, max_model_len=4096)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    QAs = load_data(data_path)
    _, answers, _ , _ = gen_sample(gen,tokenizer, QAs, 1, temperature=0.9)

    em,f1 = 0,0
    searches = 0
    failure = 0
    for item, res_answer in zip(QAs, answers):
        pattern = r"<search>(.*?)</search>"
        match = re.search(pattern, res_answer)
        if match:
            searches += 1

        pattern = r"<answer>(.*?)</answer>"
        match = re.search(pattern, res_answer)
        if match:
            res_answer = match.group(1).strip()
        else:
            failure += 1
        em += em_max_over_ground_truths(res_answer, item['answers'])
        f1 += f1_max_over_ground_truths(res_answer, item['answers'])
    print('EM:',em/len(QAs)*100, ' F1:', f1/len(QAs)*100, ' Searches:', searches, ' Failure:', failure)


if __name__ == '__main__':
    model_path = '/home/share/models/Qwen2.5-1.5B-Instruct'
    data_path = '/home/yjiang/myWork/Simple-AgenticRAG-RL/data/eval/HotpotQA.jsonl'
    main(model_path, data_path)
