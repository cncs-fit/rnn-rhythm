#%%
import tensorflow as tf
import numpy as np
import keras

from keras import layers # type:ignore
from keras.constraints import NonNeg #type:ignore

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath('')))


from models.base_model import BaseModel



class ATCLRNNModel(BaseModel):
  ''' Adaptive Time Constant Leaky RNN Model '''
  def __init__(
    self, 
    units, 
    alpha,
    N_in=1, 
    N_out=1, 
    init_state_strength=0.01,
    activation="tanh",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
    return_sequences=True,
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False,
    **kwargs,
  ):
    cell = ATCLRNNCell(
      units,
      alpha,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      dtype=kwargs.get("dtype", None),
      trainable=kwargs.get("trainable", True),
      name="simple_rnn_cell",
    )
    # Drop kwargs that would clash with positional arguments
    kwargs_filtered = {k: v for k, v in kwargs.items() if k not in ['N','N_in', 'N_out', 'activation', 'init_state_strength']}
    #print(kwargs_filtered.keys())
    super(ATCLRNNModel, self).__init__(
      cell=cell, 
      N_in=N_in, 
      N_out=N_out,
      N=units,
      activation=activation,
      init_state_strength=init_state_strength,
      return_sequences=return_sequences,
      return_state=return_state,
      go_backwards=go_backwards,
      stateful=stateful,
      unroll=unroll,
      **kwargs_filtered,
    )

  def get_initial_state(self, batch_size=None):
    ''' Generate the initial hidden state. '''
    initial_state = self.init_state_strength * tf.random.normal(shape=(batch_size, self.rnn_layer.cell.units))
    return initial_state
  

  def rnn_kernel_initial_weight(self):
    ''' Initial weights of the RNN kernel. '''
    units = self.rnn_layer.cell.units

    # Set the weights on the noise input channels to identity
    initial_weights = tf.zeros( shape=(units+self.N_in, units) ).numpy()
    initial_weights[-units:, :units] = tf.eye(units)
    # Initialize task-input weights with Xavier (Glorot) uniform
    s = tf.math.sqrt( 6 / (units + self.N_in) )
    initial_weights[:self.N_in, :] = tf.random.uniform(shape=(self.N_in, units), minval=-s, maxval=s)
    return initial_weights


  def sort_by_tau(self):
    ''' Reorder neurons in ascending order of time constant tau = 1 / alpha. '''
    alpha, W_in, W_rec, bias = self.rnn_layer.cell.get_weights()
    idx_sorted = np.argsort(1 / alpha)
    alpha = np.array(alpha)[idx_sorted]
    W_in = np.array(W_in)[:,idx_sorted]
    W_rec = np.array(W_rec)[idx_sorted][:, idx_sorted]
    bias = np.array(bias)[idx_sorted]
    self.rnn_layer.set_weights([alpha, W_in, W_rec, bias])
  

  def get_config(self):
    return super().get_config()
  

  @classmethod
  def from_config(cls, config):
    del config["dense"]

    rnn_config = config.pop("rnn_layer")
    cell_config = rnn_config.pop("cell")["config"]
    for k in ['name', 'trainable', 'dtype']:
      del rnn_config[k], cell_config[k]
    
    # Merge configs with cell_config taking precedence over rnn_config and config
    merged_config = {**config, **rnn_config, **cell_config}
    # Debug: print merged config
    # print("Merged config:", merged_config)

    return cls(**merged_config)
  

class LeakyRNNCell(layers.SimpleRNNCell):
  def __init__(self, units, alpha, **kwargs):
    super(LeakyRNNCell, self).__init__(units, **kwargs)
    self.alpha = alpha

  def call(self, inputs, states, training=None):
    prev_output = states[0] if tf.nest.is_nested(states) else states
    dp_mask = self.get_dropout_mask_for_cell(inputs, training)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(prev_output, training)
    
    
    if dp_mask is not None:
        h = tf.matmul(inputs * dp_mask, self.kernel)
    else:
        h = tf.matmul(inputs, self.kernel)
    
    if self.bias is not None:
        h = tf.add(h, self.bias)

    if rec_dp_mask is not None:
        prev_output = prev_output * rec_dp_mask

    new_state = ( 1 - self.alpha ) * prev_output + \
        self.alpha * (h + tf.matmul( self.activation(prev_output), self.recurrent_kernel))
    
    if self.activation is not None:
        output = self.activation(new_state)
    else :
       output = new_state

    new_state = [new_state] if tf.nest.is_nested(states) else new_state
    return output, new_state
  

  def get_config(self):
    config = super().get_config()
    alpha = float(self.alpha.numpy().tolist()[0])
    config.update( {'alpha': alpha} )
    return config


class ATCLRNNCell(layers.SimpleRNNCell):
    ''' Adaptive Time Constant Leaky RNN Cell '''
    def __init__(self, units, alpha, **kwargs):
        super(ATCLRNNCell, self).__init__(units, **kwargs)
        
        # initialize alpha as a trainable weight
        self.alpha = self.add_weight(
            name='alpha',
            shape=(units,),
            initializer=tf.keras.initializers.Constant(alpha),
            trainable=True,
            dtype=tf.float32,
            constraint=NonNeg()
        )

    def call(self, inputs, states, training=None):
        ''' Call method for the ATCLRNNCell.
        Args:
            inputs: Input tensor of shape (batch_size, input_dim).
            states: List of previous states, where the first element is the previous output.
            training: Boolean indicating whether the layer is in training mode.
        Returns:
            output: The output tensor of shape (batch_size, units).
            new_state: The new state tensor, which is a list containing the updated state.
        ''' 
        prev_state = states[0] if tf.nest.is_nested(states) else states
        # comment out the dropout masks for now
        # dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        # rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(prev_output, training)

        # if dp_mask is not None:
        #     h = tf.matmul(inputs * dp_mask, self.kernel)
        # else:
        h = tf.matmul(inputs, self.kernel)

        if self.bias is not None:
            h = tf.add(h, self.bias)

        # if rec_dp_mask is not None:
        #     prev_output = prev_output * rec_dp_mask

        new_state = (1 - self.alpha) * prev_state + \
            self.alpha * (h + tf.matmul(self.activation(prev_state), self.recurrent_kernel))

        # Note: activation is applied to the output but NOT to the state.
        if self.activation is not None:
            output = self.activation(new_state)
        else:
            output = new_state

        new_state = [new_state] if tf.nest.is_nested(states) else new_state
        return output, new_state

    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self.alpha.numpy().tolist()})
        return config




class RandomLeakyRNNCell(layers.SimpleRNNCell):
  def __init__(self, units, alpha, **kwargs):
    super(RandomLeakyRNNCell, self).__init__(units, **kwargs)
    
    if isinstance(alpha, (list, np.ndarray)):
        self.alpha =  tf.convert_to_tensor(alpha)
    else:
        self.alpha =  alpha
    self.alpha = tf.cast(self.alpha, dtype=tf.float32)

  def call(self, inputs, states, training=None):
    prev_output = states[0] if tf.nest.is_nested(states) else states
    dp_mask = self.get_dropout_mask_for_cell(inputs, training)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(prev_output, training)
    
    
    if dp_mask is not None:
        h = tf.matmul(inputs * dp_mask, self.kernel)
    else:
        h = tf.matmul(inputs, self.kernel)
    
    if self.bias is not None:
        h = tf.add(h, self.bias)

    if rec_dp_mask is not None:
        prev_output = prev_output * rec_dp_mask

    new_state = ( 1 - self.alpha ) * prev_output + self.alpha * (h + tf.matmul( self.activation(prev_output), self.recurrent_kernel))  #type:ignore
    
    if self.activation is not None:
        output = self.activation(new_state)
    else :
       output = new_state

    new_state = [new_state] if tf.nest.is_nested(states) else new_state
    return output, new_state

  def get_config(self):
    config = super().get_config()
    config.update( {'alpha': self.alpha.numpy().tolist(), } ) # type:ignore
    return config


class LeakyRNNModel(BaseModel):
  def __init__(
    self, 
    units, 
    alpha,
    N_in=1, 
    N_out=1, 
    N=200,
    init_state_strength=0.01,

    activation="tanh",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
      
    return_sequences=True,
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False,
    **kwargs,
  ):
    cell = LeakyRNNCell(
      units,
      alpha,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      dtype=kwargs.get("dtype", None),
      trainable=kwargs.get("trainable", True),
      name="simple_rnn_cell",
    )
    # Drop kwargs that would clash with positional arguments
    kwargs_filtered = {k: v for k, v in kwargs.items() if k not in ['N_in', 'N_out', 'N', 'activation', 'init_state_strength']}
    
    super(LeakyRNNModel, self).__init__(
      cell, 
      N_in, 
      N_out,
      N,
      activation,
      init_state_strength,
      return_sequences=return_sequences,
      return_state=return_state,
      go_backwards=go_backwards,
      stateful=stateful,
      unroll=unroll,
      **kwargs_filtered,
    )


  def get_initial_state(self, batch_size=None):
    ''' Generate the initial hidden state. '''
    initial_state = self.init_state_strength * tf.random.normal(shape=(batch_size, self.rnn_layer.cell.units))
    return initial_state
  

  def rnn_kernel_initial_weight(self):
    ''' Initial weights of the RNN kernel. '''
    units = self.rnn_layer.cell.units

    # Set the weights on the noise input channels to identity
    initial_weights = tf.zeros( shape=(units+self.N_in, units) ).numpy()
    initial_weights[-units:, :units] = tf.eye(units)
    # Initialize task-input weights with Xavier (Glorot) uniform
    s = tf.math.sqrt( 6 / (units + self.N_in) )
    initial_weights[:self.N_in, :] = tf.random.uniform(shape=(self.N_in, units), minval=-s, maxval=s)

    return initial_weights
  

  def get_config(self):
    return super().get_config()
  

  @classmethod
  def from_config(cls, config):
    del config["dense"]

    rnn_config = config.pop("rnn_layer")
    cell_config = rnn_config.pop("cell")["config"]
    for k in ['name', 'trainable', 'dtype']:
      del rnn_config[k], cell_config[k]
    
    # Merge configs with cell_config taking precedence over rnn_config and config
    merged_config = {**config, **rnn_config, **cell_config}
    
    return cls(**merged_config)


class RandomLeakyRNNModel(BaseModel):
  def __init__(
    self, 
    units, 
    alpha,
    N_in=1, 
    N_out=1, 
    N=100,
    init_state_strength=0.01,
    activation="tanh",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    kernel_constraint=None,
    recurrent_constraint=None,
    bias_constraint=None,
    dropout=0.0,
    recurrent_dropout=0.0,
      
    return_sequences=True,
    return_state=False,
    go_backwards=False,
    stateful=False,
    unroll=False,
    **kwargs,
  ):
    cell = RandomLeakyRNNCell(
      units,
      alpha,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      recurrent_initializer=recurrent_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      recurrent_regularizer=recurrent_regularizer,
      bias_regularizer=bias_regularizer,
      kernel_constraint=kernel_constraint,
      recurrent_constraint=recurrent_constraint,
      bias_constraint=bias_constraint,
      dropout=dropout,
      recurrent_dropout=recurrent_dropout,
      dtype=kwargs.get("dtype", None),
      trainable=kwargs.get("trainable", True),
      name="simple_rnn_cell",
    )
    super(RandomLeakyRNNModel, self).__init__(
      cell, 
      N_in, 
      N_out,
      N,
      activation,
      init_state_strength,
      return_sequences=return_sequences,
      return_state=return_state,
      go_backwards=go_backwards,
      stateful=stateful,
      unroll=unroll,
      **kwargs,
    )


  def get_initial_state(self, batch_size=None):
    ''' Generate the initial hidden state. '''
    initial_state = self.init_state_strength * tf.random.normal(shape=(batch_size, self.rnn_layer.cell.units))
    return initial_state
  

  def rnn_kernel_initial_weight(self):
    ''' Initial weights of the RNN kernel.

    Because the input dimension includes the noise channels in addition to
    the task input, the standard initializer cannot be applied directly,
    so the weights are set manually here.
    '''
    units = self.rnn_layer.cell.units

    # Set the weights on the noise input channels to the identity matrix
    initial_weights = tf.zeros( shape=(units+self.N_in, units) ).numpy()
    initial_weights[-units:, :units] = tf.eye(units)
    # Initialize task-input weights with Xavier (Glorot) uniform
    s = tf.math.sqrt( 6 / (units + self.N_in) )
    initial_weights[:self.N_in, :] = tf.random.uniform(shape=(self.N_in, units), minval=-s, maxval=s)

    return initial_weights
  

  def get_config(self):
    return super().get_config()
  

  @classmethod
  def from_config(cls, config):
    del config["dense"]

    rnn_config = config.pop("rnn_layer")
    cell_config = rnn_config.pop("cell")["config"]
    for k in ['name', 'trainable', 'dtype']:
      del rnn_config[k], cell_config[k]
    
    # Merge configs with cell_config taking precedence over rnn_config and config
    merged_config = {**config, **rnn_config, **cell_config}
    
    return cls(**merged_config)



# %%

if __name__ == "__main__":
  units = 100
  N_in = 1
  model = ATCLRNNModel(units, N_in, N_out = 1)

  inputs = tf.random.normal(shape=(10, 1000, N_in))
  noise = tf.random.normal(shape=(10, 1000, units)) #type:ignore
  y,ave = model(inputs, noise)

  # %%

  # import tensorflow as tf
  import matplotlib.pyplot as plt

  plt.plot(ave[0,:,0])


  # cell = RandomLeakyRNNCell(100, 0.15, 0.05)

  #inputs = tf.zeros((10,  1000,2))
  #state = tf.ones(shape=(1,100))
  # cell.build(1)

  # output, new_state = cell.call(inputs, state)

  # print(output.shape)
  # print(new_state.shape)

  # plt.hist(cell.alpha)
  # plt.show()

  # plt.hist(new_state[0,:])
  # plt.show()

  # %%
