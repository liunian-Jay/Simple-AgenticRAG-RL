
class Dict2Obj:
    def __init__(self, d):
        self.__dict__.update(d)

cfg_dict = {
    "gen_device": 1,
    "model_path": "/home/share/models/Qwen2.5-1.5B-Instruct",
    "data_path": "/home/yjiang/myWork/Simple-AgenticRAG-RL/data/hotpot_5k.jsonl",

    # GRPO 训练超参
    "beta": 0.04,
    "clip_param": 0.2,
    "all_steps": 1000,
    "save_steps": 50,
    "gen_update_steps": 8,

    # 采样超参数
    "Q_batch_size": 1,        # 一个batch，即几个question
    "num_pre_Q": 8,           # 一个问题采样的次数，应大于group
    "train_batch_size": 8 ,   # 一个训练group，应不大于num_pre_Q
    "compute_gen_logps": True,

    # 服务地址
    "ref_server_url": "http://localhost:59875"
}
train_config = Dict2Obj(cfg_dict)



ds_config = {
    "train_micro_batch_size_per_gpu": train_config.train_batch_size,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}