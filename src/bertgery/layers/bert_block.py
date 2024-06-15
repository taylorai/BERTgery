# block with both fused MLP and flash SDPA
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertLayer as HFBertLayer
from .sdpa import BertSDPA
import torch.nn as nn
try:
    from flash_attn.ops.fused_dense import FusedMLP
except ImportError as e:
    raise ImportError("FusedMLP not available. Please install flash_attn.") from e


class BertLayer(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.attention = BertSDPA(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        # weights of fused mlp are called:
        # self.fc1 = nn.Linear(in_features, hidden_features, bias=bias1, **factory_kwargs)
        # self.fc2 = nn.Linear(hidden_features, out_features, bias=bias2, **factory_kwargs)
        self.mlp = FusedMLP(
            in_features=config.hidden_size, 
            hidden_features=config.intermediate_size,
            out_features=config.hidden_size,
            bias1=True,
            bias2=True, 
            activation='gelu_approx'
        )
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

    @classmethod
    def from_hf_bert_layer(cls, layer: HFBertLayer, config: BertConfig):
        new_layer = cls(config)
        new_layer.attention = BertSDPA.from_bert_attention(config, layer.attention.self)
        new_layer.norm1 = layer.attention.output.LayerNorm
        new_layer.dropout1 = layer.attention.output.dropout
        new_layer.mlp.fc1 = layer.intermediate.dense
        new_layer.mlp.fc2 = layer.output.dense
        new_layer.norm2 = layer.output.LayerNorm
        new_layer.dropout2 = layer.output.dropout
        del layer
        return new_layer
    
    def to_hf_bert_layer(self, config: BertConfig):
        layer = HFBertLayer(config)
        layer.attention.self = self.attention.to_bert_attention()
        layer.attention.output.LayerNorm = self.norm1
        layer.attention.output.dropout = self.dropout1
        layer.intermediate.dense = self.mlp.fc1
        layer.output.dense = self.mlp.fc2
        layer.output.LayerNorm = self.norm2
        layer.output.dropout = self.dropout2
        return layer
