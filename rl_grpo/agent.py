import os, random, re, requests, time, json
from vllm import LLM, SamplingParams
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

def retrieve(query, topk=1):
    # return '小明的真实性名叫杨明'
    while True:
        try:
            response = requests.post(
                "http://localhost:8000/retrieve",
                json={"query": query, "topk": topk, "return_scores": False}, timeout=5  # 可设置超时，避免卡死
            )
            if response.status_code == 200:
                data = response.json()
                result_list = data.get('result', [])
                lines = []
                for item in result_list:
                    line = f"{item['title']}: {item['text']}"
                    lines.append(line)
                result_str = '\n\n'.join(lines)
                return result_str
            else:
                print(f"Request failed, status: {response.status_code}, retrying...")
        except Exception as e:
            print(f"Request error: {e}, retrying...")
        time.sleep(1)  # 间隔1秒再试

def gen_sample(vllm_gen, tokenizer, inputs, num_pre_Q):
    system_prompt = "You are a helpful assistant. "
    answer_prompt = """
    Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. 
    After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>, and it will return the top searched results between <information> and </information>. 
    You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer> without detailed illustrations. 
    For example, <answer> xxx </answer>. The answer, \"xxx\", should be a few short words. Question: {question}."""
    def build_prompt(query):
        return tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": answer_prompt.format(question=query)}
        ], tokenize=False, add_generation_prompt=True)

    queries = [item["Q"] for item in inputs]
    prompts = [build_prompt(query) for query in queries]

    # 多采样，每个prompt要采样 num_pre_Q 个
    n = num_pre_Q
    max_loops = 3

    # 存储每条采样的token和mask（与生成同步）
    generations = [ [p] * n for p in prompts ] 
    finished = [ [False] * n for _ in prompts ]
    search_cnt = [ [0] * n for _ in prompts ]
    all_answers = [ [''] * n for _ in prompts ]
    all_token_ids = [ [[] for _ in range(n)] for _ in prompts ] 
    all_token_masks = [ [[] for _ in range(n)] for _ in prompts ]

    while not all(all(all_finished) for all_finished in finished):
        batch_inputs = []
        mapping = []
        for i, gens in enumerate(generations):
            for j, cur_prompt in enumerate(gens):
                if not finished[i][j]  and search_cnt[i][j] < max_loops:
                    batch_inputs.append(cur_prompt)
                    mapping.append((i, j))
                elif search_cnt[i][j] >= max_loops:
                    finished[i][j] = True

        if not batch_inputs:
            break

        sampling_params = SamplingParams(n=1, temperature=0.9, max_tokens=1024,stop=["</search>"], skip_special_tokens=False)
        outputs_list = vllm_gen.generate(batch_inputs, sampling_params, use_tqdm=False)

        for k, outputs in enumerate(outputs_list):
            text = outputs.outputs[0].text
            i, j = mapping[k]

            # 前面停止词
            if outputs.outputs[0].finish_reason == "stop" and outputs.outputs[0].stop_reason == '</search>':
                text += '</search>'

            # 增量生成的部分
            generations[i][j] += text
            all_answers[i][j] += text
            new_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).tolist()
            all_token_ids[i][j] += new_ids
            all_token_masks[i][j] += [1]*len(new_ids)  

            if '</search>' in text:
                # 找到<query>片段，插入doc再继续生成
                m = re.search(r'<search>(.*?)</search>', text, re.DOTALL)# 用正则或分割提取<search>xxx</search>
                if m:
                    query_str = m.group(1).strip()
                    doc = retrieve(query_str, 1) # 检索doc
                    doc = f"<information>{doc.strip()}</information>"
                    generations[i][j] += doc
                    all_answers[i][j] += doc
                    new_ids = tokenizer(doc, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0).tolist()
                    all_token_ids[i][j] += new_ids
                    all_token_masks[i][j] += [0]*len(new_ids)
                    search_cnt[i][j] += 1
            else:
                finished[i][j] = True # stop但是无query，是直接生成完毕的情况
    
    # 扁平化结果，保持原顺序
    answers = [ans for group in all_answers for ans in group]
    ans_token_ids = [token_ids for group in all_token_ids for token_ids in group]
    ans_masks = [token_masks for group in all_token_masks for token_masks in group]
    return prompts * n, answers, ans_token_ids, ans_masks





# from transformers import AutoTokenizer
# # model_path = '/home/share/models/Qwen2.5-3B-Instruct'
# model_path = '/home/yjiang/myWork/Simple-AgenticRAG-RL/rl_grpo/step_200'
# gen = LLM(model= model_path, gpu_memory_utilization=0.85, max_model_len=1024)
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# # QAs = [
# #     {'Q':'what is python', 'A':'xx'},
# #     {'Q':'what is English', 'A':'xx'},
# # ]
# def load_data(data_path='/home/yjiang/myWork/Simple-AgenticRAG-RL/data/hotpot_5k.jsonl'):
#     with open(data_path, 'r') as f:
#         QAs = [json.loads(line) for line in f]
#     return QAs
# QAs = random.sample(load_data(),3)
# _, answers, ans_token_ids ,ans_masks = gen_sample(gen,tokenizer, QAs, 8)

# import torch
# from torch.nn.utils.rnn import pad_sequence
# tensor_ans_ids = [torch.tensor(ans_ids) for ans_ids in ans_token_ids]
# output_ids = pad_sequence(tensor_ans_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
# tensor_ans_masks = [torch.tensor(masks) for masks in ans_masks]
# mask_ids = pad_sequence(tensor_ans_masks, batch_first=True, padding_value=0)

# print(output_ids.shape)
# print(mask_ids.shape)

# for i in range(len(ans_masks)):
#     print(answers[i])
#     print('-'*100)
#     for idx, mask_val in enumerate(ans_masks[i]):
#         if mask_val == 0:
#             token_id = ans_token_ids[i][idx]
#             token_str = tokenizer.decode([token_id])
#             print(f"{token_str}", end='')
#     print('')
#     print('x'*100)