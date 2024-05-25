import torch
import torch.nn as nn
from transformers import BertModel

def extend_positions(model: BertModel, max_position: int):
    old_position_embeddings = model.embeddings.position_embeddings
    new_position_embeddings = nn.Embedding(
        max_position, 
        old_position_embeddings.weight.size(1)
    )
    avg_old_embedding = old_position_embeddings.weight.mean(dim=0)

    # copy over old positions
    new_position_embeddings.weight.data[:old_position_embeddings.weight.size(0)] = \
        old_position_embeddings.weight.data

    # initialize new positions to the average
    new_position_embeddings.weight.data[old_position_embeddings.weight.size(0):] = \
        avg_old_embedding
    
    model.embeddings.position_embeddings = new_position_embeddings
    model.config.max_position_embeddings = max_position
    model.embeddings.position_ids = torch.arange(max_position).unsqueeze(0)

