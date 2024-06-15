from ..layers.sdpa import BertSDPA
from ..layers.bert_block import BertLayer
from transformers import BertModel

def patch_attention(bert_model: BertModel):
    """
    Patch the attention mechanism of a BERT model with a custom attention mechanism.
    This modifies the model in place.
    """
    for layer in bert_model.encoder.layer:
        layer.attention.self = BertSDPA.from_bert_attention(
            bert_model.config,
            layer.attention.self
        )

def patch_layers(bert_model: BertModel):
    """
    Patch the layers of a BERT model with custom layers.
    This modifies the model in place. If you do this, it also includes
    patching attention, so don't do both.
    """
    for i, layer in enumerate(bert_model.encoder.layer):
        bert_model.encoder.layer[i] = BertLayer.from_hf_bert_layer(
            layer,
            bert_model.config
        )

def unpatch_layers(bert_model: BertModel):
    """
    Unpatch the layers of a BERT model with custom layers.
    This modifies the model in place. If you do this, it also includes
    unpatching attention, so don't do both.
    """
    for i in range(len(bert_model.encoder.layer)):
        new_layer = bert_model.encoder.layer[i].to_hf_bert_layer(bert_model.config)
        bert_model.encoder.layer[i] = new_layer