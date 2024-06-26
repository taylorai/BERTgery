import torch
import torch.nn as nn
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertSelfAttention
from typing import Optional, Tuple

class BertSDPA(nn.Module):
    def __init__(
        self, 
        config: BertConfig,
    ):
        super().__init__()
        self.config = config
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
        bert_attention_layer: BertSelfAttention
    ):
        config = config
        new_layer = cls(config)

        # Copy parameters
        device = next(new_layer.parameters()).device
        with torch.no_grad():
            new_layer.query.weight.copy_(bert_attention_layer.query.weight)
            new_layer.query.bias.copy_(bert_attention_layer.query.bias)
            new_layer.key.weight.copy_(bert_attention_layer.key.weight)
            new_layer.key.bias.copy_(bert_attention_layer.key.bias)
            new_layer.value.weight.copy_(bert_attention_layer.value.weight)
            new_layer.value.bias.copy_(bert_attention_layer.value.bias)
        del bert_attention_layer
        new_layer.to(device)
        return new_layer
    
    def to_bert_attention(self):
        bert_attention = BertSelfAttention(self.config)
        bert_attention.query.weight.copy_(self.query.weight)
        bert_attention.query.bias.copy_(self.query.bias)
        bert_attention.key.weight.copy_(self.key.weight)
        bert_attention.key.bias.copy_(self.key.bias)
        bert_attention.value.weight.copy_(self.value.weight)
        bert_attention.value.bias.copy_(self.value.bias)
        return bert_attention

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        reshapes tensor from (batch_size, seq_len, all_head_size) to (batch_size, num_heads, seq_len, head_size)
        for xformers, input tensors must be in format [B, M, H, K], where B is the batch size, M the sequence length,
        H is the number of heads, and K the embeding size per head
        """
        print("before transposing", x.size())
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        
        out = x.view(new_x_shape).permute(0, 2, 1, 3)  # b, h, m, k
        print("after transposing", out.size())

        return out

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
        # check the device of hidden_states and key weight/bias
        # print(hidden_states.device, self.key.weight.device)
        print(hidden_states.shape)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        # only need to expand attention mask if it's not already b, l, l
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len, seq_len), device=hidden_states.device)
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
    
class BertFusedSDPA(nn.Module):
    def __init__(
        self, 
        config: BertConfig,
    ):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # the original weights are each (hidden_dim, all_head_size. we just concatenate them)
        self.qkv = nn.Linear(config.hidden_size, 3 * self.all_head_size)
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
            query_weight = bert_attention_layer.query.weight
            key_weight = bert_attention_layer.key.weight
            value_weight = bert_attention_layer.value.weight
            qkv_weight = torch.cat([query_weight, key_weight, value_weight], dim=0)
            new_layer.qkv.weight.copy_(qkv_weight)

            query_bias = bert_attention_layer.query.bias
            key_bias = bert_attention_layer.key.bias
            value_bias = bert_attention_layer.value.bias
            qkv_bias = torch.cat([query_bias, key_bias, value_bias], dim=0)
            new_layer.qkv.bias.copy_(qkv_bias)

        del bert_attention_layer
        return new_layer


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
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        qkv = self.qkv(hidden_states).view(batch_size, seq_len, 3, self.num_attention_heads, self.attention_head_size)
        q, k, v = qkv.permute(0, 3, 2, 1, 4).unbind(2)

        # only need to expand attention mask if it's not already b, l, l
        if attention_mask.numel() < batch_size * seq_len**2:
            bias = attention_mask.view(batch_size, 1, seq_len)  # b, 1, k_len
            bias = bias.expand(-1, seq_len, -1).unsqueeze(1)  # b, q_len, k_len
        else:
            bias = attention_mask.view(batch_size, 1, seq_len, seq_len)  # b, 1, q_len, k_len
        out = nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout,
            attn_mask=bias,  # b, 1, q_len, k_len, broadcast happens automatically
        ).transpose(
            1, 2
        )  # b, m, h, k

        return out.flatten(2), None
        