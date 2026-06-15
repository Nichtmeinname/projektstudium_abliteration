import random
from abc import abstractmethod, ABC

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from code.methods.setup_device import setup_device


class BaseModel(ABC):
    def __init__(self, model_name: str, seed: int = 42, set_four_bit_quantization: bool = False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.set_grad_enabled(False)

        self.seed = seed
        self.device = setup_device()
        self.tokenizer = self._load_tokenizer(model_name)
        self.model = self._load_model(model_name, set_four_bit_quantization)

    @abstractmethod
    def _load_tokenizer(self, model_name: str) -> AutoTokenizer:
        """
        Define the AutoTokenizer for a specific model.
        :param model_name: The name of the specific model.
        :return: The tokenizer for a specific model.
        """
        pass

    @abstractmethod
    def _load_model(self, model_name: str, set_four_bit_quantization: bool) -> AutoModelForCausalLM:
        """
        Define the AutoModelForCasualLM for a specific model.
        :param model_name: The name of the specific model.
        :param set_four_bit_quantization: True, if 4 bit quantization is needed.
        :return: The AutoModelForCasualLM for a specific model.
        """
        pass

    @abstractmethod
    def generate_multiple(self, prompts: list, batch_size=32) -> list[dict[str, str]]:
        """
        Generate responses for multiple prompts.
        :param prompts: A list of prompts.
        :param batch_size: The batch size for the prompts.
        :return: A list of dicts with prompt and responses.
        """
        pass

    @abstractmethod
    def tokenize_prompt(self, prompts: list):
        pass

    @abstractmethod
    def get_eoi_toks(self):
        pass

    @abstractmethod
    def get_layers(self):
        pass

    @abstractmethod
    def get_num_of_hidden_layers(self):
        pass

    @abstractmethod
    def get_hidden_size(self):
        pass

    @abstractmethod
    def get_attn_modules(self):
        pass

    @abstractmethod
    def get_mlp_modules(self):
        pass

    @abstractmethod
    def get_refusal_toks(self):
        pass

    @abstractmethod
    def set_state_dict(self, state_dict):
        pass
