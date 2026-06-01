import math

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
        prompts : list[str]
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

            for prompt in current_batch:
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

            for prompt, output_ids, input_length in zip(
                    current_batch,
                    outputs,
                    input_lengths
            ):
                generated_tokens = output_ids[input_length:]

                response = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                ).strip()

                all_responses.append({
                    "prompt": prompt,
                    "response": response
                })

            del inputs
            del outputs
            torch.cuda.empty_cache()

        return all_responses

    def tokenize_prompt(self, prompt: str):
        formatted_prompt = QWEN_CHAT_TEMPLATE.format(instruction=prompt)
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        return inputs

    def get_eoi_toks(self):
        return self.tokenizer.encode(QWEN_CHAT_TEMPLATE.split("{instruction}")[-1])

    def get_layers(self):
        return self.model.model.layers

    def get_num_of_hidden_layers(self):
        return self.model.config.num_hidden_layers

    def get_hidden_size(self):
        return self.model.config.hidden_size
