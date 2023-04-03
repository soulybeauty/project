import tensorflow as tf
from vectorize import text_to_vector
import numpy as np;import json
import xgboost as xgb
import os


current_dir = os.path.dirname(os.path.abspath(__file__))

id_to_cls_path = os.path.join(current_dir, "id_to_cls.txt") 
bert_fc_classifier_path = os.path.join(current_dir, "bert_fc_classifier") 
xgb_model_path = os.path.join(current_dir, "xgb_model.json") 
finetuned_multilingual_BERT_epoch_10.model_path = os.path.join(current_dir, "path/to/finetuned_model") 

with open(id_to_cls_path, "r") as fp:
    # Load the dictionary from the file
    id_to_cls = json.load(fp)

modelo = tf.keras.models.load_model(bert_fc_classifier_path)

model_xgb_2 = xgb.XGBClassifier()
model_xgb_2.load_model(xgb_model_path)



########################BERT PART########################################

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the fine-tuned model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=3)
model.load_state_dict(torch.load(finetuned_multilingual_BERT_epoch_10, map_location=torch.device('cpu')))

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')


def get_output_bert(input_text):

    # Tokenize the input text   
    input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt')

    # Get the model prediction
    outputs = model(input_ids)
    logits = outputs[0]
    predicted_labels = torch.argmax(logits, dim=1)

    return predicted_labels.item()


while True:

    text = str(input('\n Write your message: '))

    embedding = text_to_vector(text)

    embedding = embedding.reshape(1,768)

    model_output_xg = model_xgb_2.predict(embedding)

    model_output_dl = modelo.predict(embedding.reshape(1,768))

    if np.max(model_output_dl) <0.6:
        print('\n Deeplearning: Hec biri')
    else:
        print('\n Deeplearning: ',id_to_cls [ str(np.argmax(model_output_dl) +1)] )

    print('\n XGBoost: ',id_to_cls[str(model_output_xg[0] + 1)] )

    print('\n BERT: ',id_to_cls[str(get_output_bert(text)+1)])

