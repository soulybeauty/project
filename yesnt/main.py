import tensorflow as tf
from vectorize import text_to_vector
import numpy as np;import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

id_to_cls_path = os.path.join(current_dir, "id_to_cls.txt") 
bert_fc_classifier_path = os.path.join(current_dir, "bert_fc_classifier") 

with open(id_to_cls_path, "r") as fp:
    # Load the dictionary from the file
    id_to_cls = json.load(fp)

modelo = tf.keras.models.load_model(bert_fc_classifier_path)


while True:

    text = str(input('\n Write your message: '))

    embedding = text_to_vector(text)

    model_output = modelo.predict(embedding.reshape(1,768))

    print('\n corrected version: ',id_to_cls [ str(np.argmax(model_output) +1)] )

