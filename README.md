# BERTgery
utilities for doing model surgery (remove layers, extend position embeddings, add flash attention, etc.)

## Roadmap
- put the flashattention patching in here from picovector package
- put the layer slicing thing in here from picovector
- implement similar API to the huggingface resize token embeddings but for position embeddings
    - this should also encapsulate the emb stuff from picovector like interpolation
- longer term maybe implement some obvious fusion stuff? e.g. QKV fusion