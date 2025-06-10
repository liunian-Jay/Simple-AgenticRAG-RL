import torch
from refer_llm.refer_client import RefClient

client = RefClient(server_url='http://localhost:59875')  # 换成你的实际服务端口
prompt_len = 1
merged_ids = torch.tensor([[2,2,3],[4,5,6]])
rewards = torch.tensor([1.0, 0.5])
doc_masks = torch.tensor([[1,0],[0,1]])
gen = torch.tensor([[1,0],[0,1]])

# result = client.upload(prompt_len, merged_ids, rewards, doc_masks, gen)
result = client.get_batch()
print('upload result:', result)