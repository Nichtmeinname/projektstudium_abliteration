import random

import numpy as np
import torch
from garak.generators.base import Generator
from transformers import AutoTokenizer, AutoModelForCausalLM

from code.methods.setup_device import setup_device


class LLMCustomGenerator(Generator):
    """
    Garak Generator to generate messages from huggingface models.
    """

    def __init__(self, model_name, seed=42):
        """
        Constructor for LLMCustomGenerator class.
        :param model_name: The model to generate responses from.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.seed = seed
        self.device = setup_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
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
