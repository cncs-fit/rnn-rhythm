## train_once.py
# Run one training trial and save the resulting model and history.
# %% 
import os
import sys
import time
import argparse
from math import sqrt

import json
import numpy as np
import tensorflow as tf
import keras  # Direct keras import
from keras import initializers
from keras.initializers import RandomNormal #type:ignore
# from tensorflow.keras import initializers #type:ignore
#keras.config.set_floatx("float32")
import matplotlib.pyplot as plt


from models import ATCLRNNModel
from dataset_generator import DatasetGenerator
import mycallbacks as mycb

# Random seed (set via --seed command-line argument)
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
args, _ = parser.parse_known_args()

if args.seed is not None:
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    print(f'Random seed set to {args.seed}')
else:
    print('No random seed specified. Results will not be reproducible.')

# hyperparameters
units = 100
alpha_0 = 0.02
learning_rate = 0.0001
Fs = 1000.0  # sampling frequency (Hz)
transient = 500  # transient period (steps) excluded from loss evaluation
train_length = 4000  # length of the signal used for loss evaluation (steps)
# Training iterations and batch size
n_training = 50000
batch_size = 50
# parameters for rhythm switching
switch_num = 3 
wait_length = 100
pulse_length = 100
noise_strength = 0.1


rhythms = {"theta": [4, 7], "alpha": [8, 13],
            "beta": [14, 29], "gamma": [30, 50]}

result_path = os.getcwd() + '/results/'
save_dir = f'{result_path}/'
env_dir = f'{save_dir}/env'
checkpoint_dir = f'{save_dir}/checkpoints'

# Limit GPU memory use
if __name__ == '__main__':
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
      try:
          # Allocate GPU memory dynamically rather than all at once
          for gpu in gpus:
              tf.config.experimental.set_memory_growth(gpu, True)
      except RuntimeError as e:
          print(e)
  # # Alternative GPU memory setup (some machines need this form to avoid Keras errors)
  # physical_devices = tf.config.experimental.list_physical_devices('GPU')
  # if len(physical_devices) > 0:
  #   for k in range(len(physical_devices)):
  #     tf.config.experimental.set_memory_growth(physical_devices[k], True)
  #     print('memory growth:', tf.config.experimental.get_memory_growth(
  #         physical_devices[k]))
  # else:
  #   print("Not enough GPU hardware devices available")


  # %%
  # Build the model
  model = ATCLRNNModel(units, alpha=alpha_0, N_in=len(rhythms), N_out=units,
                        recurrent_initializer=RandomNormal(mean=0, stddev=1/sqrt(units)))

  # Save the model configuration
  model.save_config(f'{env_dir}/model_config.json')

  # Compile the model
  model.compile(optimizer=tf.keras.optimizers.Adam(
      learning_rate=learning_rate,global_clipnorm=1.0))
  model.save_compile_config(f'{env_dir}/compile_config.json')

  #%%
  # Build the dataset generator

  task_config = {
      'rhythms': rhythms,
      'period_length': transient + train_length,
      'switch_num': switch_num,
      'wait_length': wait_length,
      'pulse_length': pulse_length,
  }
  # setting noise
  noise_config = {'strength': noise_strength, 'dim': units}
  # construct dataset generator
  dsg = DatasetGenerator(Fs, transient, task_config, noise_config)
  dsg.save_config(f'{env_dir}/dsg_config.json')

  # Use tf.data.Dataset with prefetching to overlap data generation and training
  dummy_inputs, dummy_noise, dummy_mask, dummy_onehots, _ = dsg.make_datasets(batch_size)
    
  def generator():
      while True:
          inputs, noise, mask, onehots, _ = dsg.make_datasets(batch_size)
          yield inputs, noise, mask, onehots

  train_dataset = tf.data.Dataset.from_generator(
      generator,
      output_signature=(
          tf.TensorSpec(shape=dummy_inputs.shape, dtype=tf.float32),
          tf.TensorSpec(shape=dummy_noise.shape, dtype=tf.float32),
          tf.TensorSpec(shape=dummy_mask.shape, dtype=tf.float32),
          tf.TensorSpec(shape=dummy_onehots.shape, dtype=tf.float32),
      )
  )
  train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
  train_iterator = iter(train_dataset)

  # TODOs:
      # - verify behavior when loading a saved model and resuming inference
      # - verify loading the training history for plotting and analysis


  earlystopping_cb = mycb.GoalBasedStopping(
      filepath=checkpoint_dir + '/best_model.weights.h5', goal=0.000001)
  # History callback
  history_cb = mycb.History(save_dir + '/history.json')
  # Callback that periodically saves model weights
  save_weights_cb = mycb.ModelCheckpoint(
    filepath=checkpoint_dir + '/weights-{epoch:04d}.weights.h5',
    save_every=500,  # save every 500 epochs
    start_from_epoch=0
  )

  # Assemble the callback list
  callbacks = mycb.CallbackList(
      [earlystopping_cb, history_cb, save_weights_cb],
      add_history=True,
      add_printlog=True,
      model=model
  )



  #%%
  # Training loop
  logs = {}
  callbacks.on_train_begin(logs)
  min_loss = np.inf
  best_weights = None

  for iter in range(n_training):
    if model.stop_training:
        break

    callbacks.on_epoch_begin(epoch=iter, logs=logs)

    # Fetch the next batch
    inputs, noise, mask, onehots = next(train_iterator)

    init_state = model.get_initial_state(batch_size)
    # Training step

    logs = model.train_step(inputs, noise, mask, onehots, init_state, #type:ignore
                            dsg.task.wait_length, transient, Fs, task_config['period_length'])
    
    if logs['loss'] < min_loss:
        min_loss = logs['loss']
        best_weights = model.get_weights()
        # Periodically persist the best weights to disk (crash safety / progress check)
    # Saving every 1000 epochs to keep the overhead small
    if iter > 1000 and iter % 1000 == 0:
        model.save_weights(f'{checkpoint_dir}/best_model.weights.h5', overwrite=True)

    callbacks.on_epoch_end(epoch=iter, logs=logs)
    
    # If stop_training was requested, save the best weights and break out of the loop
    if model.stop_training:
        if best_weights is not None:
            model.set_weights(best_weights)
            model.save_weights(f'{checkpoint_dir}/best_model.weights.h5', overwrite=True)
            print(f"Best model saved at {checkpoint_dir}/best_model.weights.h5")
        break

  callbacks.on_train_end(logs)

  # Final save of the best model
  if best_weights is not None:
      model.set_weights(best_weights)
      model.save_weights(f'{checkpoint_dir}/best_model.weights.h5', overwrite=True)

  # %%