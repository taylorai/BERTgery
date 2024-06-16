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
]).run_commands([
    'cd flash-attention/csrc/layer_norm && pip install .',
]).pip_install('bertgery@git+https://github.com/taylorai/BERTgery.git@1ab76fb')

app = modal.App('test-bertgery')

@app.function(
    image=image,
    gpu=gpu.A100()
)
def test_bertgery():
    import time
    import torch
    import tqdm
    print("torch version:", torch.__version__)
    from bertgery.layers.flash_bert import (
        convert_bertmodel_to_flash_attn_bert,
        convert_flash_attn_bert_to_bertmodel
    )
    from transformers import BertForPreTraining, BertModel, BertConfig
    model = BertModel.from_pretrained('bert-base-uncased').to('cuda').to(torch.bfloat16)
    model.eval()
    random_input = torch.randint(0, 1000, (128, 128), device='cuda')
    batch = {
        "input_ids": random_input,
        "attention_mask": torch.ones_like(random_input, device='cuda')
    }
    start = time.time()
    for _ in tqdm.trange(400):
        output1 = model(**batch).last_hidden_state
    print("HF Bert step time:", (time.time() - start) / 400)

    new_model = convert_bertmodel_to_flash_attn_bert(model)
    del model
    del batch
    torch.cuda.empty_cache()

    new_model.to(torch.bfloat16)
    new_model.to('cuda')
    new_model.eval()
    batch = {
        "input_ids": random_input,
        "attention_mask": torch.ones_like(random_input, device='cuda').to(torch.bool)
    }
    start = time.time()
    for _ in tqdm.trange(400):
        output2 = new_model(**batch).last_hidden_state
    print("Flash-Attn Bert step time:", (time.time() - start) / 400)

    time.sleep(1)
    # print first few elements of the outputs
    print(output1[0, :5, :5])
    print(output2[0, :5, :5])
    
    print("converting back to hf bert")
    new_model = convert_flash_attn_bert_to_bertmodel(new_model)

    print("conversion successful")