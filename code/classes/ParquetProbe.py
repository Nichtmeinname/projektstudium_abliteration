import time

from garak.probes.base import Probe
from transformers import AutoTokenizer


class ParquetProbe(Probe):
    """
    A probe that generates responses based on a list of prompts.
    """

    def __init__(self, prompts):
        """
        Constructor for the ParquetProbe class.
        :param prompts: A list of prompts.
        """
        super().__init__()
        self.prompts = prompts

    def probe(self, generator):
        """
        This function generates responses based on a list of prompts.
        :param generator: The llm generator to generate responses from.
        :return: A dictionary of the given prompt and the generated response.
        """
        results = []
        times_per_prompt = []
        tokens = []
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

        print(f"Beginning with Prompting with Seed {generator.seed}...")
        start_all_prompts = time.time()
        num = 1
        for prompt in self.prompts:
            start_prompt = time.time()
            print(f"({num}/{len(self.prompts)}) Prompting: ", prompt)
            response = generator.generate(prompt)[0]
            results.append({
                "prompt": prompt,
                "response": response
            })
            end_prompt = time.time()

            text_tokens_prompt = tokenizer.encode(prompt)
            text_tokens_response = tokenizer.encode(response)
            tokens.append(len(text_tokens_prompt) + len(text_tokens_response))
            times_per_prompt.append(end_prompt - start_prompt)
            num += 1
        end_all_prompts = time.time()
        return results, times_per_prompt, end_all_prompts - start_all_prompts, tokens
