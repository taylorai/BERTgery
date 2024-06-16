import re
import torch
import torch.nn as nn
from typing import Union, Tuple, Optional
from transformers import (
    BertConfig, 
    BertForPreTraining as HFBertForPreTraining, 
    BertModel as HFBertModel,
    BertForSequenceClassification as HFBertForSequenceClassification
)
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
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


def update_config_for_flash_attn(config: BertConfig):
    config.use_flash_attn = True
    config.fused_bias_fc = fused_bias_fc_is_available
    config.fused_mlp = fused_mlp_is_available
    config.fused_dropout_add_ln = fused_dropout_add_ln_is_available
    # set activation to gelu_new
    config.hidden_act = "gelu_new"
    return config

def convert_bertmodel_to_flash_attn_bert(
    model: HFBertModel,
):
    """
    Convert a Hugging Face BERT model to a Flash-Attn BERT model.
    """
    # convert the model
    config = update_config_for_flash_attn(model.config)
    new_model = FlashBertModel(config)
    # add bert. to the beginning of the keys so remap_state_dict works
    remapped_state_dict = remap_state_dict({
        "bert." + k: v for k, v in model.state_dict().items()
    }, config)

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

    # substitute .gamma for .weight and .beta for .bias -- bug in original inv_remap_state_dict
    remapped_state_dict = {
        re.sub(r"\.gamma", ".weight", k): v for k, v in remapped_state_dict.items()
    }
    remapped_state_dict = {
        re.sub(r"\.beta", ".bias", k): v for k, v in remapped_state_dict.items()
    }
    print("remapped_state_dict after re.sub:", remapped_state_dict.keys())
    
    # check for keys present in one but not the other
    pretrained_keys = set(remapped_state_dict.keys())
    new_keys = set(new_model.state_dict().keys())
    print("new keys:", new_keys)
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
    hf_config = update_config_for_flash_attn(hf_config)
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

class FlashBertForSequenceClassification:
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.config = config
        self.bert = FlashBertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    @classmethod
    def from_hf_bert_for_sequence_classification(cls, model: HFBertForSequenceClassification):
        config = update_config_for_flash_attn(model.config)
        new_model = cls(config)
        new_model.bert = convert_bertmodel_to_flash_attn_bert(model.bert)
        new_model.dropout = model.dropout
        new_model.classifier = model.classifier
        return new_model
    
    def to_hf_bert_for_sequence_classification(self):
        model = HFBertForSequenceClassification(self.config)
        model.bert = convert_flash_attn_bert_to_bertmodel(self.bert)
        model.dropout = self.dropout
        model.classifier = self.classifier
        return model

    def forward(
        self,
        input_ids,
        position_ids=None,
        token_type_ids=None,
        attention_mask=None,
        **kwargs, # these are summarily ignored lol
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            masked_tokens_mask=None
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return SequenceClassifierOutput(
            loss=None,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )