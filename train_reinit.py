## train_reinit.py
# R1-Major1 additional experiment: train with different time-constant (tau) initializations.
# Independent copy of train_once.py that keeps tau LEARNABLE but changes its INITIAL value:
#   --tau-init const50    : uniform tau = 50 ms  (== original baseline, for parity checks)
#   --tau-init const200   : uniform tau = 200 ms
#   --tau-init loguniform : per-neuron tau ~ log-uniform[tau_min, tau_max] ms (default 5..300)
# Output is written to --outdir (default ./results) so the original multiple_runs/ is never touched.
import os
import argparse
from math import sqrt

import numpy as np
import tensorflow as tf
from keras.initializers import RandomNormal  # type: ignore

from models import ATCLRNNModel
from dataset_generator import DatasetGenerator
import mycallbacks as mycb

# ----------------------------- CLI -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
parser.add_argument('--tau-init', type=str, default='const50',
                    choices=['const50', 'const200', 'loguniform'],
                    help='Time-constant initialization scheme')
parser.add_argument('--tau-min', type=float, default=5.0, help='min tau (ms) for loguniform')
parser.add_argument('--tau-max', type=float, default=300.0, help='max tau (ms) for loguniform')
parser.add_argument('--outdir', type=str, default=os.getcwd() + '/results',
                    help='Directory to write results/ into')
parser.add_argument('--max-iters', type=int, default=50000,
                    help='Max training iterations (lower for smoke tests)')
parser.add_argument('--goal', type=float, default=1e-6,
                    help='Early-stopping loss goal (1e-5 justified by stability check for R1-1)')
args, _ = parser.parse_known_args()

if args.seed is not None:
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    print(f'Random seed set to {args.seed}')
else:
    print('No random seed specified. Results will not be reproducible.')

# ----------------------- hyperparameters -----------------------
units = 100
alpha_0 = 0.02  # only used to build the cell; overwritten below by --tau-init
learning_rate = 0.0001
Fs = 1000.0
transient = 500
train_length = 4000
n_training = 50000
batch_size = 50
switch_num = 3
wait_length = 100
pulse_length = 100
noise_strength = 0.1

rhythms = {"theta": [4, 7], "alpha": [8, 13], "beta": [14, 29], "gamma": [30, 50]}

save_dir = args.outdir
env_dir = f'{save_dir}/env'
checkpoint_dir = f'{save_dir}/checkpoints'


def make_alpha_init(scheme, n, tau_min, tau_max):
    """Return an (n,) array of initial leak rates alpha = 1/tau (tau in steps == ms at Fs=1000)."""
    if scheme == 'const50':
        tau = np.full(n, 50.0)
    elif scheme == 'const200':
        tau = np.full(n, 200.0)
    elif scheme == 'loguniform':
        # log-uniform in tau over [tau_min, tau_max]
        tau = np.exp(np.random.uniform(np.log(tau_min), np.log(tau_max), size=n))
    else:
        raise ValueError(scheme)
    return (1.0 / tau).astype(np.float32)


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # ----------------------- model -----------------------
    model = ATCLRNNModel(units, alpha=alpha_0, N_in=len(rhythms), N_out=units,
                         recurrent_initializer=RandomNormal(mean=0, stddev=1 / sqrt(units)))

    # --- override the per-neuron time-constant initialization ---
    alpha_init = make_alpha_init(args.tau_init, units, args.tau_min, args.tau_max)
    model.rnn_layer.cell.alpha.assign(alpha_init)
    tau_init = 1.0 / alpha_init
    print(f'[tau-init={args.tau_init}] tau(ms): min={tau_init.min():.1f} '
          f'median={np.median(tau_init):.1f} max={tau_init.max():.1f}')

    model.save_config(f'{env_dir}/model_config.json')
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=learning_rate, global_clipnorm=1.0))
    model.save_compile_config(f'{env_dir}/compile_config.json')

    # record the initialization scheme alongside the model
    with open(f'{env_dir}/tau_init_config.json', 'w') as f:
        import json
        json.dump({'tau_init': args.tau_init, 'tau_min': args.tau_min,
                   'tau_max': args.tau_max, 'seed': args.seed,
                   'alpha_init': alpha_init.tolist()}, f, indent=2)

    # ----------------------- dataset -----------------------
    task_config = {
        'rhythms': rhythms,
        'period_length': transient + train_length,
        'switch_num': switch_num,
        'wait_length': wait_length,
        'pulse_length': pulse_length,
    }
    noise_config = {'strength': noise_strength, 'dim': units}
    dsg = DatasetGenerator(Fs, transient, task_config, noise_config)
    dsg.save_config(f'{env_dir}/dsg_config.json')

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

    # ----------------------- callbacks -----------------------
    earlystopping_cb = mycb.GoalBasedStopping(
        filepath=checkpoint_dir + '/best_model.weights.h5', goal=args.goal)
    history_cb = mycb.History(save_dir + '/history.json')
    save_weights_cb = mycb.ModelCheckpoint(
        filepath=checkpoint_dir + '/weights-{epoch:04d}.weights.h5',
        save_every=500, start_from_epoch=0)

    callbacks = mycb.CallbackList(
        [earlystopping_cb, history_cb, save_weights_cb],
        add_history=True, add_printlog=True, model=model)

    # ----------------------- training loop -----------------------
    logs = {}
    callbacks.on_train_begin(logs)
    min_loss = np.inf
    best_weights = None

    for it in range(min(n_training, args.max_iters)):
        if model.stop_training:
            break
        callbacks.on_epoch_begin(epoch=it, logs=logs)
        inputs, noise, mask, onehots = next(train_iterator)
        init_state = model.get_initial_state(batch_size)
        logs = model.train_step(inputs, noise, mask, onehots, init_state,  # type: ignore
                                dsg.task.wait_length, transient, Fs, task_config['period_length'])
        if logs['loss'] < min_loss:
            min_loss = logs['loss']
            best_weights = model.get_weights()
        if it > 1000 and it % 1000 == 0:
            model.save_weights(f'{checkpoint_dir}/best_model.weights.h5', overwrite=True)
        callbacks.on_epoch_end(epoch=it, logs=logs)
        if model.stop_training:
            if best_weights is not None:
                model.set_weights(best_weights)
                model.save_weights(f'{checkpoint_dir}/best_model.weights.h5', overwrite=True)
                print(f"Best model saved at {checkpoint_dir}/best_model.weights.h5")
            break

    callbacks.on_train_end(logs)
    if best_weights is not None:
        model.set_weights(best_weights)
        model.save_weights(f'{checkpoint_dir}/best_model.weights.h5', overwrite=True)
    print(f'Done. min_loss={min_loss:.3e}  outdir={save_dir}')
