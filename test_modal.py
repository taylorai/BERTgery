import modal
from modal import gpu

image = modal.Image.from_registry('nvcr.io/nvidia/pytorch:24.05-py3').pip_install(
    'packaging',
    'ninja',
    'transformers'
).run_commands([
    'git clone https://github.com/Dao-AILab/flash-attention.git',
    'cd flash-attention && python setup.py install',
    'cd flash-attention/csrc/fused_dense_lib && pip install .'
]).pip_install('bertgery@git+https://github.com/taylorai/BERTgery.git@1312cb0')

app = modal.App('test-bertgery')

@app.function(
    image=image,
    gpu=gpu.A100()
)
def test_bertgery():
    import torch
    print("torch version:", torch.__version__)
    from bertgery.layers.flash_bert import convert_hf_model_to_flash_attn
    from transformers import BertModel, BertConfig
    model = BertModel.from_pretrained('bert-base-uncased').to('cuda')
    model.eval()
    random_input = torch.randint(0, 1000, (2, 128), device='cuda')
    batch = {
        "input_ids": random_input,
        "attention_mask": torch.ones_like(random_input, device='cuda')
    }
    output1 = model(**batch).last_hidden_state
    
    new_model = convert_hf_model_to_flash_attn(model, model.config)
    new_model.to('cuda')
    new_model.eval()
    output2 = new_model(**batch).last_hidden_state
    
    # print first few elements of the outputs
    print(output1[0, :5, :5])
    print(output2[0, :5, :5])
    assert torch.allclose(output1, output2, atol=1e-6), "Patching attention did not work"
