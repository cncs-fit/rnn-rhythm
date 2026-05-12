# %%
import os

from abc import ABC, abstractmethod
import json

import numpy as np
import tensorflow as tf
#from tensorflow import keras
import keras  # Direct keras import
from keras import layers



def to_float64(obj):
  ''' Convert a tensor or ndarray (including nested list/dict) to float64. '''
  if isinstance(obj, (tf.Tensor, tf.Variable)):
    obj = obj.numpy().astype(np.float64).tolist() # type:ignore
  elif isinstance(obj, (np.ndarray, np.floating, np.signedinteger)):
    obj = obj.astype(np.float64).tolist()
  elif isinstance(obj, list):
    for i, o in enumerate(obj):
      obj[i] = to_float64(o)
  elif isinstance(obj, dict):
    for k, v in obj.items():
      obj[k] = to_float64(v)
  return obj

def power(x, Fs):
  '''
  calculate power spectrum of signal x
  args:
    x: signal tensor, shape=(bs, t_len, N)
    Fs: sampling frequency
  '''

  x_f = tf.signal.rfft(x)[..., 1:] # remove DC component
  return (tf.math.abs(x_f)**2) / (Fs*x.shape[-1]) #

class BaseModel(tf.keras.Model, ABC):

  def __init__(
      self, 
      cell, 
      N_in=1, 
      N_out=1,
      N=400,
      activation='tanh',
      init_state_strength=0.01,
      return_sequences=True,
      return_state=False,
      go_backwards=False,
      stateful=False,
      unroll=False,
      zero_output_for_mask=False,
      **kwargs,
  ):
    super(BaseModel, self).__init__()

    self.N = N
    self.N_in = N_in
    self.N_out = N_out

    self.activation = activation
    self.act = tf.keras.activations.get('tanh')
    if activation is not None:
        self.act = tf.keras.activations.get(activation)
    
    
    self.init_state_strength = init_state_strength
    self.history = None

    # initialize RNN Layer
    self.rnn_layer = layers.RNN(
      cell, 
      return_sequences,
      return_state,
      go_backwards,
      stateful,
      unroll,
      zero_output_for_mask,
    )
    # Generate Dense Layer
    self.dense = layers.Dense(N_out) 

    self.build_network()

  def call(self, inputs, noise, initial_state=None, training=None):
    ''' forward pass of the model.
     The model has two input streams: task input and noise input.
    args:
      inputs: Task input, shape:(bs, t_len, N_in)
      noise: Noise input shape:(bs, t_len, N)
      initial_state: initial state of the hidden layer
      training: Boolean indicating whether the call is meant for training or inference
    '''

    Input = tf.concat([inputs, noise], axis=2)
    self.r = self.rnn_layer(Input, initial_state=initial_state, training=training) #(bs, t_len, N)

    average = tf.reduce_mean(self.r, axis=2, keepdims=True) # averaging over N
    self.y = self.dense(self.r)  # task output
    return self.y, average
  

  def build_network(self):
    inputs = tf.random.normal(shape=(10, self.N, self.N_in))
    noise = tf.random.normal(shape=(10, self.N, self.N)) #type:ignore
    self(inputs, noise)

    # Set initial input weights of the RNN cell
    self.rnn_layer.cell.kernel.assign( self.rnn_kernel_initial_weight() ) #type:ignore
    self.dense.kernel.assign( tf.eye(self.rnn_layer.cell.units, self.N_out) ) #type:ignore
    # Set the mask used during gradient computation
    self._set_grad_mask()
  

  @abstractmethod
  def rnn_kernel_initial_weight(self):
    ''' Return the initial weights for the RNN kernel. '''
    pass

  @abstractmethod
  def get_initial_state(self, batch_size=None):
    ''' Generate the initial hidden state. '''
    pass

  # def get_weight_index(self, weight):
  #   ''' Return the index of `weight` within self.trainable_variables. '''
  #   for i, v in enumerate(self.trainable_variables):
  #     if weight is v:
  #       return i
  #   return None

  def get_weight_index(self, weight):
    ''' Return the index of `weight` within self.trainable_variables. '''
    try:
        return self.trainable_variables.index(weight)
    except ValueError:
        return None


  def _set_grad_mask(self):
    ''' Build a mask that freezes the input weights coming from the noise channels. '''
    self.gradient_mask = [tf.ones_like(v) for v in self.trainable_variables]

    ind_input_kernel = self.get_weight_index(self.rnn_layer.cell.kernel)
    if ind_input_kernel is not None:
      input_kernel_mask = self.gradient_mask[ind_input_kernel].numpy()
      input_kernel_mask[-self.rnn_layer.cell.units:, :] = 0
      self.gradient_mask[ind_input_kernel] = tf.convert_to_tensor( input_kernel_mask )


  def _apply_grad_mask(self, grad):
    ''' Apply the gradient mask. '''

    # Replace gradients of unused layers with zeros
    grad = [tf.zeros_like(v) if g is None else g
             for v, g in zip(self.trainable_variables, grad)]

    # Apply the mask
    grad = [g * m for g, m in zip(grad, self.gradient_mask)]
    return grad


  @tf.function
  def train_step(
    self, 
    inputs, 
    noise, 
    mask, 
    onehots,
    init_state, 
    wait_length,
    transient,
    Fs,
    len_interval = 1000, # interval between each switch (ms)
    ):
    """ training step for power spectrum learning
    args:
      inputs: Task input, shape:(bs, t_len, N_in)
      noise: Noise input shape:(bs, t_len, N)
      mask: Mask for rhythm learning, shape:( N_rhythm, N_freq) 
      onehots: One-hot encoding of rhythms, shape:(bs, switch_num, N_rhythm)
      init_state: Initial state for RNN
      wait_length: Length of wait time before the first pulse
      transient: Length of transient time to be excluded from the power spectrum calculation
      Fs: Sampling frequency
    """
    switch_num = onehots.shape[1]
    
    with tf.GradientTape() as tape:
      y, z = self(inputs, noise, init_state) #run rnn # type:ignore
      z = z[:,wait_length:,0]  # drop the pre-pulse wait period
      # print(f'len_interval: {len_interval}, switch_num: {switch_num}, transient: {transient}, Fs: {Fs}')
      # print(f'z shape: {z.shape}, mask shape: {mask.shape}, onehots shape: {onehots.shape}')
      # Split the output signal into per-interval segments
      z_split = tf.reshape(z[:,:switch_num*len_interval], (z.shape[0], switch_num, -1)) #(bs, switch_num, t_len)
      # Compute the power spectrum of each interval (transient excluded)
      pz_split = power(z_split[:,:,transient:], Fs) # (bs, switch_num, N_freq)

      # Inner product of per-interval power spectrum and rhythm masks
      mask_sum = tf.matmul(pz_split, tf.transpose(mask)) # shape: (bs,switch_num, N_rhythm)
      # Total power in each interval
      whole_sum = tf.reduce_sum(pz_split, axis=-1, keepdims=True)
      # Fraction of total power within each rhythm band
      ratio = mask_sum / whole_sum #(bs, switch_num, N_rhythm)

      s = tf.cast(tf.reduce_sum(onehots, axis=-1, keepdims=True), dtype=tf.float32)  # number of active bands per interval (bs, switch_num, 1)
      rhythm_sum = tf.reduce_sum( onehots * ( (ratio - 1/s)**2 ), axis=-1 ) #type:ignore
      loss = tf.reduce_mean( rhythm_sum )

    grad = tape.gradient(loss, self.trainable_variables)
    grad = self._apply_grad_mask(grad)
    self.optimizer.apply_gradients(zip(grad, self.trainable_variables))

    logs = { 'loss': loss }
    return logs

  def get_config(self):
    config =  {
      'N': self.N,
      'N_in': self.N_in,
      'N_out': self.N_out,
      'activation': self.activation,
      'init_state_strength': self.init_state_strength,
    }
    config["rnn_layer"] = self.rnn_layer.get_config()
    config["dense"] = self.dense.get_config()
    return config


  def get_compile_config(self):
    """Get basic compile configuration"""
    config = {}
    
    if hasattr(self, 'optimizer') and self.optimizer:
        config['optimizer'] = {
            'class_name': self.optimizer.__class__.__name__,
            'config': self.optimizer.get_config()
        }
    
    return config

  def save_config(self, filepath):
    save_dir = os.path.split(filepath)[0]
    os.makedirs(save_dir, exist_ok=True)

    config = {'class_name': self.__class__.__name__, 
              'config': self.get_config()}
    with open(filepath, 'w') as f: 
      json.dump( config, f, indent=4)


  def save_compile_config(self, filepath):
    save_dir = os.path.split(filepath)[0]
    os.makedirs(save_dir, exist_ok=True)
    
    with open(filepath, 'w') as f:
      json.dump( self.get_compile_config(), f, indent=4)


  @classmethod
  def from_config(cls, config):
    cell_config = config.pop("rnn_layer")["cell"]
    cell = layers.deserialize(cell_config)
    del config["dense"]

    return cls(cell, **config)
  

  def compile_from_config(self, config):
    # Build the optimizer from the config
    opt_class_name = config['optimizer'].pop('class_name')
    optimizer_class = getattr(tf.keras.optimizers, opt_class_name)
    optimizer = optimizer_class(**config['optimizer'])

    self.compile(optimizer=optimizer)


  @classmethod
  def load(cls, filepath):
    with open(filepath, 'r') as f:
      model_config = json.load(f)
    return cls.from_config( model_config["config"] )
  

  def load_compile(self, filepath):
    with open(filepath, 'r') as f:
      compile_config = json.load(f)
    self.compile_from_config( compile_config )



# %%