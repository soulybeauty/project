from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.backend as K
import pickle

# with open(r'C:\Users\ii.karimli\Desktop\translation\Machine-translation-with-attention-model\human_vocab','rb') as fp:
#     human_vocab = pickle.load(fp)
    
# with open(r'C:\Users\ii.karimli\Desktop\translation\Machine-translation-with-attention-model\machine_vocab','rb') as fp:
#     machine_vocab = pickle.load(fp)

# len_human_vocab = len(human_vocab)
# print('human_vocab inside',len_human_vocab)
# len_machine_vocab = len(machine_vocab)
# print('machine_vocab inside',machine_vocab)

def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')



class neural_translation_model(layers.Layer):
    
    def __init__(self, Tx, Ty, n_a, n_s, human_vocab_size
                                                           , machine_vocab_size):
        
        # Default parameter for model
        self.Tx = Tx
        self.Ty = Ty        
        self.n_a = n_a # number of units for the pre-attention, bi-directional LSTM's hidden state 'a' 
        self.n_s = n_s # number of units for the post-attention LSTM's hidden state "s"
        self.human_vocab_size = human_vocab_size
        self.machine_vocab_size = machine_vocab_size
        
        
        
        # We will share weights with those layer. In order to prevent them to be intialized for each time step we can either 
        # define them as a global variable or we can create their object
        self.repeator = layers.RepeatVector(Tx)
        self.concatenator =  layers.Concatenate(axis=-1)
        self.densor1 = layers.Dense(10, activation = "tanh")
        self.densor2 = layers.Dense(1, activation = "relu")
        self.activator = layers.Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
        self.dotor = layers.Dot(axes = 1)
        
        self.post_activation_LSTM_cell = layers.LSTM(n_s, return_state = True) # Please do not modify this global variable.
        self.output_layer = layers.Dense(machine_vocab_size, activation=softmax)
        print(machine_vocab_size)
        
    def a_step_attention(self, a, s_prev):
        #it is same activation that will be shared for all t_delta activations to calculate alpha
        s_prev = self.repeator(s_prev)
        #concatenate the activations with hidden state of post attention LSTM 
        concatenation = self.concatenator([a,s_prev])
        
        #Here is the small fully connected neural network to find attention weights 
        # intermediate energies
        e = self.densor1(concatenation)
        # Energies
        energies = self.densor2(e)
        #softmax to calculate alphas
        alpha = self.activator(energies)
        
        # context = sum_over_t_x( alpha(t_y,t_x)) * a(t_x)
        context = self.dotor([alpha,a])
        
        return context
    
    def model(self):
        
        X  = layers.Input(shape = (self.Tx,self.human_vocab_size))
        s0 = layers.Input(shape = (self.n_s,), name ='s0')
        c0 = layers.Input(shape = (self.n_s,), name ='c0')
        
        s = s0 
        c = c0 
        
        a = layers.Bidirectional(layers.LSTM(self.n_a ,return_sequences= True))(X)
        
        outputs = []
        
        for t in range(self.Ty):
            
            context = self.a_step_attention(a, s)
            
            s, _, c = self.post_activation_LSTM_cell(context,initial_state=[s, c])
            
            out = self.output_layer(s)
            
            outputs.append(out)
            
        model = tf.keras.Model(inputs = [X,s0,c0] , outputs = outputs)
        
        return model
    