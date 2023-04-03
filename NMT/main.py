from modelo import neural_translation_model
from nmt_utils import *
import os 

current_dir = os.path.dirname(os.path.abspath(__file__))

weights_path = os.path.join(current_dir, "models\\58epochs_weights.h5") 

#CONFIG
n_s = 64
n_a = 32
Tx = 30
Ty = 10

#Needed dicts

with open('NMT/vocabs/human_vocab','rb') as fb:
    human_vocab = pickle.load(fb)

with open('NMT/vocabs/machine_vocab','rb') as fb:
    machine_vocab = pickle.load(fb)

with open('NMT/vocabs/inv_machine_vocab','rb') as fb:
    inv_machine_vocab = pickle.load(fb)

len_human_vocab = len(human_vocab)
len_machine_vocab = len(machine_vocab)


from modelo import neural_translation_model

attention_model = neural_translation_model(Tx = Tx, Ty = Ty, n_a = n_a, n_s = n_s, human_vocab_size = len_human_vocab
                                                           , machine_vocab_size = len_machine_vocab).model()

# print(attention_model.summary())

attention_model.load_weights(weights_path)

EXAMPLES = ['10 yanvar 2025', '21 avqust 2016', '10 iyun 2007', 'Şənbə May 9 2018', 'Mart 3 2001', '1 mart 2001','aprelin 18-də 98']


s00 = np.zeros((1, n_s))
c00 = np.zeros((1, n_s))

def get_output(example):
    example = example.lower().replace('-',' ')
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    source = np.swapaxes(source, 0, 1)
    source = np.expand_dims(source, axis=0)
    prediction = attention_model.predict([source, s00, c00])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    print("source:", example)
    print("output:", ''.join(output),"\n")

while True:
    inpt = str(input('Write the date: '))
    get_output(inpt)