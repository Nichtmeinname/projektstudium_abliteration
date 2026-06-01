from code.classes.Config import Config
from code.classes.generators.QwenLLMModel import QwenLLMModel


def select_model(config: Config):
    if "qwen" in config.model_alias.lower():
        return QwenLLMModel(model_name=config.model_path, seed=config.seed,
                            set_four_bit_quantization=config.four_bit_quantization)
    else:
        raise ValueError("No such model implemented yet. Check your model name or create an own BaseModel.")
