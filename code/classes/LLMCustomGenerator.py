import random

import numpy as np
import torch
from garak.generators.base import Generator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from code.methods.setup_device import setup_device


class LLMCustomGenerator(Generator):
    """
    Garak Generator to generate messages from huggingface models.
    """

    def __init__(self, model_name, seed=42, set_four_bit_quantisation=False):
        """
        Constructor for LLMCustomGenerator class.
        :param seed: The seed to use.
        :param set_four_bit_quantisation: True, if 4 Bit Quantization is needed. False otherwise.
        :param model_name: The model to generate responses from.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 4-bit Quantisierung konfigurieren
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.seed = seed
        self.device = setup_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config if set_four_bit_quantisation else None,
            dtype=torch.float16,
            device_map=self.device
        )

        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()
        torch.set_grad_enabled(False)

    def generate(self, prompt, **kwargs):
        """
        Generate responses from models with a given prompt using torch and transformers library.
        :param prompt: The prompt to generate responses from.
        :param kwargs: The keyword arguments to use for generating the responses.
        :return: The response from the prompt.
        """
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.7,
                do_sample=True
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        # GPU Cleanup
        del outputs
        del inputs
        del generated_tokens

        torch.cuda.empty_cache()

        return response
