# BERTgery
utilities for doing model surgery (remove layers, extend position embeddings, add flash attention, etc.)

## Roadmap
- put the flashattention patching in here from picovector package
- put the layer slicing thing in here from picovector
- implement similar API to the huggingface resize token embeddings but for position embeddings
    - this should also encapsulate the emb stuff from picovector like interpolation
- longer term maybe implement some obvious fusion stuff? e.g. QKV fusion


do everything here: https://bookface.ycombinator.com/posts/80527
https://github.com/Dao-AILab/flash-attention/tree/22339db185027324f334a7f59e2584da266bfd4c/training

pad and unpad: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/models/bert.py
cramming stuff: https://github.com/andersonbcdefg/cramBERT