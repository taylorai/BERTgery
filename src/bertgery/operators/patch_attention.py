from ..layers.sdpa import BertSDPA
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