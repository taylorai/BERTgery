from typing import Union
from transformers import BertConfig, BertModel as HFBertModel
try:
    from flash_attn.models.bert import BertModel as FlashAttnBertModel, remap_state_dict
except ImportError as e:
    raise ImportError("Flash-Attn not available. Please install flash_attn.") from e


def convert_hf_model_to_flash_attn(model: HFBertModel, config: BertConfig) -> FlashAttnBertModel:
    """
    Convert a Hugging Face BERT model to a Flash-Attn BERT model.
    """
    new_model = FlashAttnBertModel(config)
    new_model.load_state_dict(
        remap_state_dict(model.state_dict(), config), 
        strict=False
    )
    return new_model