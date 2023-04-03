import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
# import torch

# Load the pre-trained BERT model and tokenizer
model_name = "azizbarank/mbert-finetuned-azerbaijani-ner"
tokenizer_transform = AutoTokenizer.from_pretrained(model_name)
model_embeddings = AutoModel.from_pretrained(model_name)


def text_to_vector(input_text, embedding_model = model_embeddings ,transformer_tokenizer = tokenizer_transform):
    tokens = transformer_tokenizer.encode(input_text, add_special_tokens=True)
    if len(tokens) > 512:
        tokens = tokens[:512]
    input_ids = torch.tensor([tokens])
    with torch.no_grad():
        outputs = embedding_model(input_ids)
        embeddings = outputs[0][0][1:-1].mean(dim=0)
    return np.array(embeddings)

# text = 'xeyr'

# print(text_to_vector(text))