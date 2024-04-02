# Get Low-Rank Model
fp16 = True
max_rank = 128
device = 'cuda'

LOW_RANK_CONFIG = {
    "q_proj": {
        "max_rank": max_rank,
    },
    "k_proj": {
        "max_rank": max_rank,
    },
    "v_proj": {
        "max_rank": max_rank,
    },
    "o_proj": {
        "max_rank": max_rank,
    },
    "up_proj": {
        "max_rank": max_rank,
    },
    "down_proj": {
        "max_rank": max_rank,
    },
    "gate_proj": {
        "max_rank": max_rank,
    },
    "common": {
        "fp16": fp16,
        "device": device,
    },
}