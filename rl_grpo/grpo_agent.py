import os, random, time, json
from tqdm import tqdm
from queue import Empty

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams


from refer_llm.refer_client import RefClient
from reward import reward_EM_F1
from agent import gen_sample
from config import train_config, ds_config
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def load_data(data_path='/home/yjiang/myWork/Simple-AgenticRAG-RL/data/hotpot_5k.jsonl'):
    with open(data_path, 'r') as f:
        QAs = [json.loads(line) for line in f]
    return QAs


def GRPO_step(batch, engine, train_config):
    def get_per_token_logps(logits, input_ids):
        per_token_logps = [] # Use a loop to reduce memory peak.
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
        #from kernel.ce_kernel import fast_log_softmax_gather
        #get_per_token_logps = fast_log_softmax_gather

    prompt_length = batch['prompt_len']
    inputs = batch['inputs'].to(engine.device)
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:,prompt_length-1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
    
    # modified by JAY
    if 'doc_masks' in batch:
        # batch['doc_masks'] 形状 [B, L-gen]，如果不是的话，则说明某个地方代码仍然存在BUG
        mask = batch['doc_masks'].to(completion_mask.device)  
        completion_mask = completion_mask * mask

    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        clipped_ratio = torch.clamp(ratio, 1-train_config.clip_param, 1+train_config.clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else: 
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert train_config.compute_gen_logps is False
    per_token_loss = -(per_token_loss - train_config.beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    return loss


def get_reward(inputs, answers, num_pre_Q):
    rewards = []
    for i, inp in enumerate(inputs):
        for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
            rewards.append(reward_EM_F1(inp, a))
    return torch.tensor(rewards, dtype=torch.float32)
    

def process_one(vllm_gen, prompt, answers, ans_token_ids, rewards, doc_masks, refClient, train_config):
    """处理一个prompt下的所有答案, 得到更新所需的所有数据"""
    
    def generator_logps(vllm_gen, merged_ids, prompt_len):
        gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)
        outputs = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
        logps_list = [output.prompt_logprobs[prompt_len:] for output in outputs]
        gen_logps = torch.tensor([[list(logprob.values())[0].logprob for logprob in logprobs] for logprobs in logps_list])
        return gen_logps

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    prompt_len = prompt_ids.shape[1]

    batch_size = train_config.train_batch_size
    for batch_start in range(0, len(answers), batch_size):
        # 取当前 group
        batch_answers = answers[batch_start:batch_start+batch_size]
        batch_ans_ids = ans_token_ids[batch_start:batch_start+batch_size]
        batch_rewards = rewards[batch_start:batch_start+batch_size]
        # if batch_rewards.max() - batch_rewards.min() < 1e-4: continue # 跳过奖励差别小的group
        if batch_rewards.max() - batch_rewards.min() == 0 : continue # 跳过奖励无差别的group
        batch_rewards = (batch_rewards - batch_rewards.mean()) / (batch_rewards.std() + 1e-4)

        # 正常情况padding后是和res部分形状相同
        batch_doc_masks = doc_masks[batch_start:batch_start+batch_size]
        tensor_doc_masks = [torch.tensor(doc_masks) for doc_masks in batch_doc_masks]
        tensor_doc_masks = pad_sequence(tensor_doc_masks, batch_first=True, padding_value=1)

        # prompt + res 合并，用于请求ref_server
        tensor_ans_ids = [torch.tensor(ans_ids) for ans_ids in batch_ans_ids]
        output_ids = pad_sequence(tensor_ans_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        prompt_ids = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, prompt_len)
        merged_ids = torch.cat([prompt_ids, output_ids], dim=1)

        gen_logps = None
        if train_config.compute_gen_logps:
            gen_logps = generator_logps(vllm_gen, merged_ids, prompt_len)
        res = refClient.upload(prompt_len, merged_ids, batch_rewards, tensor_doc_masks, gen_logps)
        # print('upload result:', res)
        if not res:
            print('上传失败')

def gen_worker(gen_Queue, train_config):

    def try_update_model():
        try:
            new_state_dict = gen_Queue.get_nowait()
            print('[VLLM PROC] recving new model ...')
            llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(new_state_dict.items())
            print('[VLLM PROC] model updated')
            del new_state_dict
        except Empty:
            print(f"\033[33mempty\033[0m")
            return
        except Exception as e:
            print(f"\033[31m{e}\033[0m")
            print('\033[32m[VLLM PROC] no new model!\033[0m')
            return

    # 初始化环境
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{train_config.gen_device}'
    print(f"Generation worker process uses GPU {train_config.gen_device}")
    torch.cuda.set_device(0)
    # 初始化模型和数据
    vllm_gen = LLM(model=train_config.model_path, gpu_memory_utilization=0.65)
    # assert tokenizer.vocab_size == vllm_gen.get_tokenizer().vocab_size
    QAs = load_data(train_config.data_path) # 加载数据集

    # print(QAs[0])
    print(vllm_gen)
    # 开始训练
    for it in range(999999999):
        if it % 3 == 0: 
            try_update_model()
        # 准备数据,并采样回复
        inputs = random.sample(QAs, train_config.Q_batch_size)
        tic = time.time()
        prompts, answers, ans_token_ids, ans_masks = gen_sample(vllm_gen, tokenizer, inputs, train_config.num_pre_Q)
        rewards = get_reward(inputs, answers, train_config.num_pre_Q)
        mean_reward = rewards.mean().item()
        print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards)
        print(f'\033[36mMean reward: {mean_reward}\033[0m')
        if it % 5 == 0: 
            print('answers:', answers[rewards.argmax().item()])
    
        # 对每个prompt分别处理上传处理
        for i, prompt in enumerate(prompts):
            start = i * train_config.num_pre_Q
            end = (i + 1) * train_config.num_pre_Q

            cur_answers = answers[start:end]
            cur_ans_ids = ans_token_ids[start:end]
            cur_rewards = rewards[start:end]
            cur_doc_masks = ans_masks[start:end]
            process_one(vllm_gen, prompt, cur_answers, cur_ans_ids, cur_rewards, cur_doc_masks, refClient, train_config)


tokenizer = AutoTokenizer.from_pretrained(train_config.model_path)
refClient = RefClient(train_config.ref_server_url)
if __name__ == '__main__':
    import deepspeed
    deepspeed.init_distributed()

    if dist.get_rank() == 0:
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn')
        gen_Queue = mp.Queue()
        p = mp.Process(target=gen_worker, args=(gen_Queue, train_config))
        p.start()

    model = AutoModelForCausalLM.from_pretrained(train_config.model_path, torch_dtype=torch.bfloat16, _attn_implementation="sdpa")
    engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, model_parameters=model.parameters())

    progress = range(1, train_config.all_steps+1)
    if dist.get_rank() == 0: progress = tqdm(progress)
    for step in progress:
        batch = refClient.get_batch()
        while batch is None:
            print('waiting for batch...'); 
            time.sleep(1)
            batch = refClient.get_batch()
        
        loss = GRPO_step(batch, engine, train_config)
        engine.backward(loss)
        engine.step()

        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")

        if step % train_config.gen_update_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('[TRAINING PROC] sending latest state_dict ...')
                state_dict = engine.module.state_dict()
                gen_Queue.put(state_dict)
                print('[TRAINING PROC] send state_dict ok!')
            dist.barrier()

        if step % train_config.save_steps == 0:
            dist.barrier()
            if dist.get_rank() == 0:
                print('saving model')
                save_name = f"./step_{step}"
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier()