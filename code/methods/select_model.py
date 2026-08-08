from code.classes.Config import Config
from code.classes.generators.Qwen25Model import Qwen25Model


def select_model(config: Config, abliteration_process: bool = False):
    if "qwen" in config.model_alias.lower():
        return Qwen25Model(model_name=config.model_path, seed=config.seed,
                           set_four_bit_quantization=config.four_bit_quantization if not abliteration_process else False)
    else:
        raise ValueError("No such model implemented yet. Check your model name or create an own BaseModel.")
