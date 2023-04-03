import numpy as np 
import pickle ; import os
from nmt_utils import *


# it gets sample that we wrote and saved as written_data inside datasets folder and enhance it with using faker by "number_of_needed_sample"
number_of_needed_sample = 9000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(number_of_needed_sample)


# Create the directory if it doesn't exist
if not os.path.exists('NMT/vocabs'):
    os.makedirs('NMT/vocabs')

# Define the file paths for saving the dictionaries
human_vocab_path = os.path.join('NMT/vocabs', 'human_vocab')
machine_vocab_path = os.path.join('NMT/vocabs', 'machine_vocab')
inv_machine_vocab_path = os.path.join('NMT/vocabs', 'inv_machine_vocab')

# Create the directory if it doesn't exist
if not os.path.exists('NMT/datasets'):
    os.makedirs('NMT/datasets')

#path for dataset created by faker+writter data
data_path = os.path.join('NMT/datasets', 'dataset')

NUM_OF_SAMPLES = len(dataset)

print(f"Number of total samples: {NUM_OF_SAMPLES}")


# save dictionaries
with open(data_path,'wb') as fb:
    pickle.dump(dataset,fb)
    print('Dataset saved succesfully!')

with open(human_vocab_path,'wb') as fb:
    print('human_vocab: \n',human_vocab)
    pickle.dump(human_vocab,fb)
    print('human_vocab saved succesfully!')

with open(machine_vocab_path,'wb') as fb:
    print('machine_vocab: \n', machine_vocab)
    pickle.dump(machine_vocab,fb)
    print('machine_vocab saved succesfully!')

with open(inv_machine_vocab_path,'wb') as fb:
    print('inv_machine_vocab: \n',inv_machine_vocab)
    pickle.dump(inv_machine_vocab,fb)
    print('inv_machine_vocab saved succesfully!')



