import gc
import math
import os

import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from code.classes.generators.BaseModel import BaseModel

SAMPLE_SYSTEM_PROMPT = """You are a helpful assistant."""

QWEN_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_REFUSAL_TOKS = [40, 2121]  # ['I', 'As']


class QwenLLMModel(BaseModel):
    """
    Qwen2.5 model from huggingface models.
    """

    def _load_tokenizer(self, model_name: str):
        return AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left"
        )

    def _load_model(self, model_name: str, set_four_bit_quantization: bool):
        # 4-bit Quantisierung konfigurieren
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config if set_four_bit_quantization else None,
            dtype=torch.float16,
            device_map=self.device
        ).eval()

        model.generation_config.pad_token_id = self.tokenizer.eos_token_id
        model.requires_grad_(False)
        return model

    def generate_multiple(self, prompts: list, batch_size=32) -> list[dict[str, str]]:
        """
        Generate responses for multiple prompts using batched inference.

        Parameters
        ----------
        prompts : list[str] or List[Dict[str, str]]
            List of prompts to process.

        batch_size : int
            Number of prompts processed simultaneously.

        Returns
        -------
        list[dict]
        """

        all_responses = []
        n_batches = math.ceil(len(prompts) / batch_size)
        num_of_current_batch = 1

        print(f"Beginning with Prompting with Seed {self.seed} using batch size {batch_size}...")
        for i in range(0, len(prompts), batch_size):

            current_batch = prompts[i:i + batch_size]
            print(f"({num_of_current_batch}/{n_batches}) Aktueller Batch:")
            num_of_current_batch += 1

            texts = []
            categories = []

            for prompt in current_batch:
                if type(prompt) == dict:
                    prompt, category = prompt["instruction"], prompt["category"]
                    categories.append(category)
                print(f"    Prompt: {prompt}")
                messages = [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]

                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                texts.append(text)

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            input_lengths = [
                len(ids)
                for ids in inputs["input_ids"]
            ]

            with torch.inference_mode():

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            if len(categories) == 0:
                categories = ["null"] * batch_size

            for prompt, output_ids, input_length, category in zip(
                    current_batch,
                    outputs,
                    input_lengths,
                    categories
            ):
                generated_tokens = output_ids[input_length:]

                response = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                ).strip()

                all_responses.append({
                    "prompt": prompt["instruction"] if type(prompt) == dict else prompt,
                    "response": response,
                    "category": category
                })

            del inputs
            del outputs
            torch.cuda.empty_cache()

        return all_responses

    def tokenize_prompts(self, prompts: list):
        formatted_prompts = [QWEN_CHAT_TEMPLATE.format(instruction=prompt) for prompt in prompts]
        inputs = self.tokenizer(
            formatted_prompts,
            padding=True,
            truncation=False,
            return_tensors="pt"
        ).to(self.device)
        return inputs

    def get_eoi_toks(self):
        return self.tokenizer.encode(QWEN_CHAT_TEMPLATE.split("{instruction}")[-1])

    def get_layers(self):
        return self.model.model.layers

    def get_num_of_hidden_layers(self):
        return self.model.config.num_hidden_layers

    def get_hidden_size(self):
        return self.model.config.hidden_size

    def get_attn_modules(self):
        return torch.nn.ModuleList([layer.self_attn for layer in self.get_layers()])

    def get_mlp_modules(self):
        return torch.nn.ModuleList([layer.mlp for layer in self.get_layers()])

    def get_refusal_toks(self):
        return [40, 2121]

    def set_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def get_attn_q_proj_weight(self, layer):
        return self.get_layers()[layer].self_attn.q_proj.weight

    def set_attn_q_proj_weight(self, weight, layer):
        self.get_layers()[layer].self_attn.q_proj.weight = weight

    def get_mlp_down_proj_weight(self, layer):
        return self.get_layers()[layer].mlp.down_proj.weight

    def set_mlp_down_proj_weight(self, weight, layer):
        self.get_layers()[layer].mlp.down_proj.weight = weight

    def save_model(self, model_file: str):
        self.model.save_pretrained(model_file, save_compressed=True)

    def load_model(self, model_file: str, set_four_bit_quantization: bool = False):
        if torch.cuda.is_available():
            print(f"    Memory (VRAM) before deleting model: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        else:
            ram = psutil.Process(os.getpid())
            print(f"    RAM used before loading model: {ram.memory_info().rss / 1024 ** 3:.2f} GB")

        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        self.model = self._load_model(model_file, set_four_bit_quantization)

        if torch.cuda.is_available():
            print(f"    Memory (VRAM) after loading model: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
        else:
            ram = psutil.Process(os.getpid())
            print(f"    RAM used after loading model: {ram.memory_info().rss / 1024 ** 3:.2f} GB")
