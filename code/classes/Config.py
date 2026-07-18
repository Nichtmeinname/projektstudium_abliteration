from dataclasses import dataclass


@dataclass
class Config:
    model_alias: str
    model_path: str
    n_train: int = 128
    n_test: int = 572
    n_val: int = 32
    seed: int = 42
    four_bit_quantization: bool = False
    batch_size: int = 32
