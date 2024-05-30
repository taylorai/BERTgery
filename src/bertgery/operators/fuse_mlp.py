# # https://github.com/Dao-AILab/flash-attention/blob/22339db185027324f334a7f59e2584da266bfd4c/flash_attn/ops/fused_dense.py

# try:
#     from flash_attn.ops.fused_dense import FusedMLP
# except ImportError:
#     print("FusedMLP not available. Please install flash_attn.")
#     FusedMLP = None

# def patch_mlp(model: BertModel):