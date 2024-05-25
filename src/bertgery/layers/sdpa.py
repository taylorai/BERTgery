import torch
import torch.nn as nn
from transformers import BertConfig
from typing import Optional, Tuple

class BertSDPA(nn.Module):
    def __init__(
        self, 
        config: BertConfig,
    ):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = config.attention_probs_dropout_prob
        self.position_embedding_type = "absolute"
        self.max_position_embeddings = config.max_position_embeddings
        self.is_decoder = False

    @classmethod
    def from_bert_attention(
        cls,
        config,
        bert_attention_layer
    ):
        config = config
        new_layer = cls(config)

        # Copy parameters
        with torch.no_grad():
            new_layer.query.weight.copy_(bert_attention_layer.query.weight)
            new_layer.query.bias.copy_(bert_attention_layer.query.bias)
            new_layer.key.weight.copy_(bert_attention_layer.key.weight)
            new_layer.key.bias.copy_(bert_attention_layer.key.bias)
            new_layer.value.weight.copy_(bert_attention_layer.value.weight)
            new_layer.value.bias.copy_(bert_attention_layer.value.bias)

        del bert_attention_layer
        return new_layer

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        reshapes tensor from (batch_size, seq_len, all_head_size) to (batch_size, num_heads, seq_len, head_size)
        for xformers, input tensors must be in format [B, M, H, K], where B is the batch size, M the sequence length,
        H is the number of heads, and K the embeding size per head
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        
        return x.view(new_x_shape).permute(0, 2, 1, 3)  # b, h, m, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # do separate linear layers for qkv if not merged
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        # only need to expand attention mask if it's not already b, l, l
        if attention_mask.numel() < batch_size * seq_len**2:
            bias = attention_mask.view(batch_size, 1, seq_len)  # b, 1, k_len
            bias = bias.expand(-1, seq_len, -1).unsqueeze(1)  # b, q_len, k_len
        else:
            bias = attention_mask.view(batch_size, 1, seq_len, seq_len)  # b, 1, q_len, k_len
        out = nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            dropout_p=self.dropout,
            attn_mask=bias,  # b, 1, q_len, k_len, broadcast happens automatically
        ).transpose(
            1, 2
        )  # b, m, h, k

        return out.flatten(2), None