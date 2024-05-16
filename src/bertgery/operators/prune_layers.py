import copy
from transformers import BertModel, BertConfig

# i just came up with this but i think it basically always works
def select_layers(num_src_layers, num_dst_layers):
    if num_dst_layers == 1:
        return [0]
    keep_every = (num_src_layers - 1) / (num_dst_layers - 1)
    layers_to_keep = range(num_dst_layers)
    layers_to_keep = [int(i * keep_every) for i in layers_to_keep]
    layers_to_keep[-1] = num_src_layers - 1
    # make sure all items are unique
    assert len(layers_to_keep) == len(set(layers_to_keep)), "duplicate layers!"
    return layers_to_keep

def prune_layers(bert_model: BertModel, layers_to_keep: list[int]):
    """
    Prune the layers of a BERT model by removing all layers not in layers_to_keep.
    Technically you could also use this to repeat layers?
    """
    small_config: BertConfig = copy.deepcopy(bert_model.config)
    small_config.num_hidden_layers = len(layers_to_keep)
    small_model = BertModel(small_config)

    # instead of copying weights, why not just assign layers to new model
    for new_layer_idx, old_layer_idx in enumerate(layers_to_keep):
        small_model.encoder.layer[new_layer_idx] = copy.deepcopy(
            bert_model.encoder.layer[old_layer_idx]
        )
    
    del bert_model
    return small_model