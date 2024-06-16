from typing import Union
from transformers import BertConfig, BertForPreTraining as HFBertForPreTraining, BertModel as HFBertModel
try:
    from flash_attn.models.bert import BertForPreTraining as FlashAttnBertModel, remap_state_dict
except ImportError as e:
    raise ImportError("Flash-Attn not available. Please install flash_attn.") from e
try:
    from flash_attn.ops.fused_dense import FusedMLP, FusedDense
    fused_mlp_is_available = True
    fused_bias_fc_is_available = True
except ImportError as e:
    fused_mlp_is_available = False
    fused_bias_fc_is_available = False
try:
    from flash_attn.ops.layer_norm import DropoutAddLayerNorm
    fused_dropout_add_ln_is_available = True
except ImportError as e:
    fused_dropout_add_ln_is_available = False

def load_flash_attn_bert(
    model_name_or_path: str,
    return_only_bert: bool = False,
) -> FlashAttnBertModel:
    """
    Load a Hugging Face BERT model and return a Flash-Attn BERT model.
    """
    hf_config: BertConfig = BertConfig.from_pretrained(model_name_or_path)
    hf_model = HFBertForPreTraining.from_pretrained(model_name_or_path)
    hf_config.use_flash_attn = True
    hf_config.fused_bias_fc = fused_bias_fc_is_available
    hf_config.fused_mlp = fused_mlp_is_available
    hf_config.fused_dropout_add_ln = fused_dropout_add_ln_is_available
    # set activation to gelu_new
    hf_config.hidden_act = "gelu_new"
    new_model = FlashAttnBertModel(hf_config)
    pretrained_state_dict = hf_model.state_dict()
    remapped_state_dict = remap_state_dict(pretrained_state_dict, hf_config)

    # check for keys present in one but not the other
    pretrained_keys = set(remapped_state_dict.keys())
    new_keys = set(new_model.state_dict().keys())
    missing_from_pretrained = new_keys - pretrained_keys
    print("WARNING: Missing keys from pretrained model:", missing_from_pretrained)
    
    new_model.load_state_dict(remapped_state_dict, strict=False)
    if return_only_bert:
        new_model = new_model.bert
    
    return new_model

def save_flash_attn_bert():
    pass