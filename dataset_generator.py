# %%
import os
import json

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class SwitchRhythm():
  def __init__(
    self, 
    rhythms={'alpha': (8,13), 'beta': (14,30), 'gamma':(30, 50)}, 
    period_length=1000, 
    switch_num=3, 
    wait_length=100,  # number of steps before the first pulse
    pulse_length=20,
    Fs=1000.0,
    transient=0
  ):
    self.rhythms = rhythms
    self.period_length = period_length
    self.switch_num = switch_num  # number of intervals
    self.wait_length = wait_length
    self.pulse_length = pulse_length
    self._Fs = Fs
    self._transient = transient  # steps at the beginning of each interval that are excluded while the network settles into the new state

    self.rhythm_names = np.array( list( self.rhythms.keys() ) )
    self.rhythm_ranges = np.array( list(self.rhythms.values()) )
    self.n_rhythm = len( self.rhythms )
    self.rhythm_to_num = {k:i for i,k in enumerate(rhythms.keys())}
    self.num_to_rhythm = {i:k for i,k in enumerate(rhythms.keys())}
    self.generate_mask()  

  @property
  def data_length(self):
    return self.wait_length + self.switch_num * self.period_length
  
  def generate_mask(self):
    ''' generate a mask for rhythm learning
    The mask is a binary matrix where each row corresponds to a rhythm and each column corresponds to a frequency bin.
    The value is 1 if the rhythm is present in that frequency bin, otherwise 0.
    The frequency bins are calculated based on the period length and the sampling frequency.
    The mask is used to select the frequency bins that correspond to the rhythms.
    '''
    # mask: (n_rhythm, len(f_ax))
    def find_index_ranges(f_ax, f_range):
      """
      f_ax: frequency axis
      f_range: frequency ranges for each rhythm
      Returns the start and end indices for each frequency range in f_ax.
      """
      index_range = np.zeros_like(f_range) # (n_rhythm, 2)
      for i, (start, end) in enumerate(f_range):
        start_idx = np.where(f_ax >= start)[0][0]
        end_idx = np.where(f_ax >= end)[0][0]
        index_range[i] = [start_idx, end_idx]
      return index_range
    
    train_len = self.period_length - self._transient
    f_ax = np.arange(1, 1+train_len//2, dtype=np.float32) * (self._Fs/train_len)
    f_range = np.array( list(self.rhythms.values()) )
    idx_range = find_index_ranges(f_ax, f_range)
    mask = np.zeros(shape=(self.n_rhythm, len(f_ax)))
    for i in range(self.n_rhythm):
      s, e = idx_range[i]
      mask[i, s:e] = 1
    self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)

  def make_datasets(self, batch_size, specified_rhythm=[]):
    ''' make datasets for rhythm learning
    args:
      batch_size: number of samples in a batch
      specified_rhythm: list of rhythm names to be specified in the dataset. If empty, random rhythms are used.
    return:
      inputs: input tensor of shape (batch_size, data_length, n_rhythm)
      mask: mask tensor of shape (n_rhythm, len(f_ax))
      onehots: one-hot encoded tensor of shape (batch_size, switch_num, n_rhythm)
      labels: labels tensor of shape (batch_size, switch_num) 


    '''
    # seeds: (batch_size, switch_num)
    specified_len = len(specified_rhythm)
    
    # Choose a random rhythm for each  interval
    randoms = np.random.randint(0,self.n_rhythm, size=(batch_size, self.switch_num-specified_len))
    specified_indx = [ self.rhythm_to_num[rhythm] for rhythm in specified_rhythm ]
    rhythm_indx = np.concatenate([[specified_indx]*batch_size, randoms], axis=1)
    vec_num_to_rhythm = np.vectorize(lambda x: self.num_to_rhythm[x])
    # labels
    labels = np.array(vec_num_to_rhythm(rhythm_indx).tolist()) # label sequences (batch_size, switch_num)

    # onehots: (batch_size, switch_num, n_rhythm) 
    num_to_onehot = {i:np.eye(self.n_rhythm, dtype='int')[i].tolist() for i in range(self.n_rhythm)}
    vec_map_value = np.vectorize(lambda x: num_to_onehot[x], otypes=[np.ndarray])
    onehots = np.array(vec_map_value(rhythm_indx).tolist())
    onehots = tf.convert_to_tensor( onehots, dtype=tf.float32)

    # inputs: (batch_size, data_length, n_rhythm)
    blank_len = self.period_length - self.pulse_length
    waits = np.zeros(shape=(batch_size, self.wait_length, self.n_rhythm))
    pulses = np.repeat(onehots[:,:,np.newaxis,:], self.pulse_length, axis=2) # (batch_size, switch_num, pulse_length, n_rhythm)
    blanks = np.zeros(shape=(batch_size, self.switch_num, blank_len, self.n_rhythm))

    pulse_interval = np.concatenate([pulses, blanks], axis=2) # (batch_size, switch_num, period_length, n_rhythm)
    pulse_interval = np.reshape(pulse_interval, newshape=(batch_size, self.switch_num * self.period_length, self.n_rhythm))
    inputs = np.concatenate([waits, pulse_interval], axis=1)
    inputs = tf.convert_to_tensor( inputs, dtype=tf.float32 )
    
    return inputs, self.mask, onehots, labels
  
  def get_config(self):
    config = {
      'rhythms': self.rhythms,
      'period_length': self.period_length,
      'switch_num': self.switch_num,
      'wait_length': self.wait_length,
      'pulse_length': self.pulse_length,
    }
    return config
  
  @classmethod
  def from_config(cls, config):
    return cls(**config)


class NoiseGenerator():
  def __init__(self, strength=0.1, dim=1):
    self.strength = strength
    self.dim = dim

  def generate(self, batch_size, data_length):
    random = np.random.normal(0, 1, size=(batch_size, data_length, self.dim))
    return tf.convert_to_tensor(self.strength * random, dtype=tf.float32)

  def plot_example(self, data_length):
    noise = self.generate(1, data_length)
    for i in range( noise.shape[-1] ):
      plt.plot( noise[0, :, i] )
    plt.title("noise")
    plt.show()

  def get_config(self):
    return {'strength': self.strength, 'dim': self.dim}

  @classmethod
  def from_config(cls, config):
    return cls(**config)


class DatasetGenerator():
  def __init__(
      self, 
      Fs=1000.0,
      transient=0,
      task_config={}, 
      noise_config={}
    ):
    self.transient = transient
    self.Fs = Fs
    #print(f'period_length: task_config["period_length"]')
    self.task = SwitchRhythm(Fs=Fs, transient=transient, **task_config)
    #print(self.task.get_config())
    self.noise_generator = NoiseGenerator(**noise_config)

  @property
  def data_length(self):
    return self.task.data_length
  
  def update_task_config(self, **kwargs):
    """
      rhythms: dict mapping rhythm name to (low, high) frequency band (Hz).
      period_length: length of one interval (steps).
      switch_num: number of rhythm switches per trial.
      wait_length: number of steps before the first pulse.
      pulse_length: length of the input pulse (steps).
    """
    task_config = self.task.get_config()
    task_config.update(kwargs)
    self.task = SwitchRhythm(Fs=self.Fs, transient=self.transient, **task_config)

  def update_noise_config(self, **kwargs):
    """
      strength: noise amplitude (sigma).
      dim: dimensionality of the noise channel.
    """
    noise_config = self.noise_generator.get_config()
    noise_config.update(kwargs)
    self.noise_generator = NoiseGenerator(**noise_config)

  def make_datasets(self, batch_size, specified_rhythm=[]):
    """
    args:
      batch_size

    return:
      inputs: input tensor
      noise: noise tensor
      mask: frequency-band mask
      onehots: one-hot encoded rhythm labels
      labels: rhythm name labels
    """
    inputs, mask, onehots, labels = self.task.make_datasets(batch_size, specified_rhythm)
    noise = self.noise_generator.generate(batch_size, self.data_length)
    return inputs, noise, mask, onehots, labels
  
  # def plot_example(self, rhythm_arr=[]):
  #   self.task.plot_example(rhythm_arr)
  #   self.noise_generator.plot_example(self.data_lengthgth)

  def get_config(self):
    config = {
      'data_length': self.data_length,
      'transient': self.transient, 
      'Fs': self.Fs,
      'task_config': self.task.get_config(),
      'noise_config': self.noise_generator.get_config()
    }
    return config

  @classmethod
  def from_config(cls, config):
    del config['data_length']
    return cls(**config)
  
  def save_config(self, filepath):
    save_dir = os.path.split(filepath)[0]
    os.makedirs(save_dir, exist_ok=True)

    with open(filepath, 'w') as f: 
      json.dump( self.get_config(), f, indent=4)

  @classmethod
  def from_json(cls, filepath):
    with open(filepath, 'r') as f:
      config = json.load(f)
    return cls.from_config( config )
  

# %%

if __name__ == '__main__':
  batch_size = 10
  transient = 0
  Fs = 1000.0
  task_config = {
    'rhythms': {'alpha': (8,13), 'beta': (14,30), 'gamma':(30, 50)}, 
    'period_length': 1000, 
    'switch_num': 10, 
    'wait_length': 100, 
    'pulse_length': 20,
  }
  noise_config = {'strength': 0.1, 'dim': 10}
  
  dsg = DatasetGenerator( Fs, transient, task_config, noise_config )

  inputs, noise, mask, onehots, labels = dsg.make_datasets(batch_size=batch_size,specified_rhythm=["alpha","beta","gamma"])
  print(f'{inputs.shape=}')
  print(f'{noise.shape=}')
  print(f'{mask.shape=}')
  print(f'{onehots.shape=}')
  print(f'{labels.shape=}')
  
  # Update task configuration
  # dsg.update_task_config(period_length=500)

  # Get configuration
  # dsg_config = dsg.get_config()

  # Build a DatasetGenerator from a config dict
  # dsg_from_config = DatasetGenerator.from_config(dsg_config)

  # Save configuration to a JSON file
  # dsg.save_config('./dsg_config_test.json')

  # Build a DatasetGenerator from a JSON file
  # dsg_from_json = dsg.from_json('./dsg_config_test.json')

  # %%
  plt.plot(inputs[0,:,:])
  plt.show()
  plt.title("inputs")
  plt.plot(mask.numpy().T)
  plt.title("mask")
  plt.show()
  plt.plot(noise[0,:,:])
  plt.title("noise")
  plt.show()
  plt.plot(onehots[0,:,:])
  plt.title("onehots")
  plt.show()
  plt.imshow(onehots[0,:,:].numpy().T, aspect='auto')
  plt.show()

  print("labels:", labels[0,:]) #?



  #rhythms={'alpha': (8,13), 'beta': (14,30), 'gamma':(30, 50)}

# %%
