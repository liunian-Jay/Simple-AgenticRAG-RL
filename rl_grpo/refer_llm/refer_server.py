import json, os
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM

import bottle, threading, queue
from bottle import request, Bottle



from refer_llm.tensor_utils import *


# =========================
# 推理相关
# =========================
def get_per_token_logps(model, input_ids: Tensor) -> Tensor:
    """
    返回每个 token 的 log prob（忽略第一个 token，对应下一个 token 的预测）
    """
    with torch.inference_mode():
        logits = model(input_ids).logits  # (B, L, Vocab)
        logits = logits[:, :-1, :]       # (B, L-1, Vocab)
        input_ids = input_ids[:, 1:]     # (B, L-1)
        log_probs = logits.log_softmax(dim=-1)
        # 取 input_ids 在各行的概率
        token_log_prob = torch.gather(log_probs, dim=2, index=input_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_prob  # (B, L-1)


# =========================
# 服务器初始化
# =========================
def create_app(raw_queue, result_queue):
    app = Bottle()

    @app.route('/upload', method='POST')
    def do_upload():
        data_list = bytes_list_to_list(request.body.read())
        data = {
            'meta': json.loads(data_list[0]),
            'inputs': bytes_to_tensor(data_list[1]),
            'rewards': bytes_to_tensor(data_list[2]),
            'doc_masks': bytes_to_tensor(data_list[3])
        }
        if len(data_list) == 5:
            data['gen_logps'] = bytes_to_tensor(data_list[4])
        raw_queue.put(data)
        return b"ok"

    @app.route('/get', method='GET')
    def do_get():
        if result_queue.empty(): return b'empty'
        return result_queue.get()

    return app

# =========================
# 推理模型循环
# =========================
def process_loop(ref_model, raw_queue, result_queue):
    while True:
        try:
            data = raw_queue.get()
            prompt_length = data['meta']['prompt_len']
            per_token_logps = get_per_token_logps(ref_model, data['inputs'].to(ref_model.device))
            # 只保留 prompt 之后的 token
            per_token_logps = per_token_logps[:, prompt_length-1:]
            out_data = [
                json.dumps(data['meta']).encode(),
                tensor_to_bytes(data['inputs']),
                tensor_to_bytes(data['rewards']),
                tensor_to_bytes(data['doc_masks']),
                tensor_to_bytes(per_token_logps)
            ]
            if 'gen_logps' in data:
                # print('返回了gen_logps')
                out_data.append(tensor_to_bytes(data['gen_logps']))
            xdata = make_bytes_list(out_data)
            result_queue.put(xdata)
        except Exception as e:
            print(f"[Process Loop Error]: {e}")

def main():
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    model_path = "/home/share/models/Qwen2.5-3B-Instruct"

    print("Loading model...")
    ref_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
    ref_model.eval()
    ref_model.requires_grad_(False)

    raw_queue = queue.LifoQueue()
    result_queue = queue.LifoQueue()
    # raw_queue = queue.Queue()
    # result_queue = queue.Queue()
    app = create_app(raw_queue, result_queue)

    # 用线程分别跑 web 和 推理主循环
    threading.Thread(target=lambda: bottle.run(app, host='0.0.0.0', port=59875, server='tornado'), daemon=True).start()
    process_loop(ref_model, raw_queue, result_queue)

if __name__ == '__main__':
    main()