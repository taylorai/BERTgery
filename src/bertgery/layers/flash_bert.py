import re
from typing import Union
from transformers import BertConfig, BertForPreTraining as HFBertForPreTraining, BertModel as HFBertModel
try:
    from flash_attn.models.bert import (
        BertForPreTraining as FlashBertForPreTraining, 
        BertModel as FlashBertModel,
        remap_state_dict,
        inv_remap_state_dict
    )
except ImportError as e:
    raise ImportError("Flash-Attn not available. Please install flash_attn.") from e
try:
    from flash_attn.ops.fused_dense import FusedMLP, FusedDense
    fused_mlp_is_available = True
    fused_bias_fc_is_available = True
    print("Fused dense kernels are available.")
except ImportError as e:
    fused_mlp_is_available = False
    fused_bias_fc_is_available = False
try:
    from flash_attn.ops.layer_norm import DropoutAddLayerNorm
    fused_dropout_add_ln_is_available = True
    print("Fused dropout-add-layer-norm kernels are available.")
except ImportError as e:
    fused_dropout_add_ln_is_available = False

# TODO: alternative version of remapping state dict that lets you do the sequence classification etc.
# BERT shapes.
def convert_bertmodel_to_flash_attn_bert(
    model: HFBertModel,
):
    """
    Convert a Hugging Face BERT model to a Flash-Attn BERT model.
    """
    # convert the model
    hf_config = model.config
    hf_config.use_flash_attn = True
    hf_config.fused_bias_fc = fused_bias_fc_is_available
    hf_config.fused_mlp = fused_mlp_is_available
    hf_config.fused_dropout_add_ln = fused_dropout_add_ln_is_available
    # set activation to gelu_new
    hf_config.hidden_act = "gelu_new"

    new_model = FlashBertModel(hf_config)
    # add bert. to the beginning of the keys so remap_state_dict works
    remapped_state_dict = remap_state_dict({
        "bert." + k: v for k, v in model.state_dict().items()
    }, model.config)

    # edit the state dict to remove the 'bert' from the beginning of the keys
    remapped_state_dict = {
        re.sub(r"^bert\.", "", k): v for k, v in remapped_state_dict.items()
    }
    
    # check for keys present in one but not the other
    pretrained_keys = set(remapped_state_dict.keys())
    new_keys = set(new_model.state_dict().keys())
    missing_from_pretrained = new_keys - pretrained_keys
    print("WARNING: Missing keys from pretrained model:", missing_from_pretrained)
    new_model.load_state_dict(remapped_state_dict)
    return new_model

def convert_flash_attn_bert_to_bertmodel(
    model: FlashBertModel,
):
    """
    Convert a Flash-Attn BERT model to a Hugging Face BERT model.
    """
    # convert the model
    hf_config = model.config
    # set activation to gelu -- probably shouldn't do this if model was finetuned with gelu_new?
    # hf_config.hidden_act = "gelu"
    new_model = HFBertModel(hf_config)
    
    # add bert. to the beginning of the keys so remap_state_dict works
    remapped_state_dict = inv_remap_state_dict({
        "bert." + k: v for k, v in model.state_dict().items()
    }, model.config)
    print("remapped_state_dict:", remapped_state_dict.keys())

    # edit the state dict to remove the 'bert' from the beginning of the keys
    remapped_state_dict = {
        re.sub(r"^bert\.", "", k): v for k, v in remapped_state_dict.items()
    }
    
    # check for keys present in one but not the other
    pretrained_keys = set(remapped_state_dict.keys())
    new_keys = set(new_model.state_dict().keys())
    missing_from_pretrained = new_keys - pretrained_keys
    print("WARNING: Missing keys from pretrained model:", missing_from_pretrained)
    
    new_model.load_state_dict(remapped_state_dict)
    return new_model

def load_flash_attn_bert(
    model_name_or_path: str,
    return_only_bert: bool = False,
):
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