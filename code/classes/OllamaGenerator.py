from garak.generators.base import Generator
from ollama import chat


class OllamaGenerator(Generator):
    """
    Garak Generator to generate messages from Ollama models.
    """

    def __init__(self, model="richardyoung/zephyr-7b-beta-abliterated:Q4_K_M", seed=42):
        """
        Constructor for Ollama Generator class.
        :param model: The model to generate responses from.
        """
        self.model = model
        self.seed = seed

    def generate(self, prompt, **kwargs):
        """
        Generate responses from Ollama models with a given prompt.
        :param prompt: The prompt to generate responses from.
        :param kwargs: The keyword arguments to use for generating the responses.
        :return: A list of the response, because garak wants a list.
        """
        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                # Fix seed, low temperature and max generated tokens to ensure reproducibility.
                "seed": self.seed,
                "temperature": 0.0,
                "num_predict": 400
            }
        )
        return [response.message.content]
