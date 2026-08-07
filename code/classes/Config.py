from dataclasses import dataclass


@dataclass
class Config:
    model_alias: str
    model_path: str
    n_train: int = 128
    n_test: int = 572
    n_val: int = 32
    seed: int = 42
    four_bit_quantization: bool = True
    batch_size: int = 32
    modify_self_attn_q_proj = False
    modify_self_attn_k_proj = False
    modify_self_attn_v_proj = False
    modify_self_attn_o_proj = True
    modify_mlp_gate_proj = False
    modify_mlp_up_proj = False
    modify_mlp_down_proj = True
