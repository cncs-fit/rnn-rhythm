import os
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import seaborn as sns
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import models
from dataset_generator import DatasetGenerator

#  ------------------------ Computation helpers ------------------------

def power(x, Fs):
    x_f = tf.signal.rfft(x)[..., 1:]
    return (tf.math.abs(x_f)**2) / (Fs*x.shape[-1])


def split_signal(x, wait_length, switch_num):
    period_length = (x.shape[-1]-wait_length) // switch_num
    x_cut_wait = x[..., wait_length:]
    return np.array( [ x_cut_wait[..., i*period_length:(i+1)*period_length] for i in range(switch_num) ] )


def calc_axis(data_len, period_len, Fs):
    t_ax = np.arange(data_len) / Fs
    f_ax = np.arange(1, 1+period_len//2, dtype=np.float32)*(Fs/period_len)
    return t_ax, f_ax


def calc_amplitudes(data, cut):
    """Return the time-averaged Hilbert envelope amplitude (legacy name: peak-to-peak)."""
    s, e = cut
    analytic = signal.hilbert(data, axis=-1)
    envelope = np.abs(analytic)
    return np.mean(envelope[..., s:e], axis=-1)


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band') #type:ignore
    filtered_data = signal.filtfilt(b, a, data, axis=-1)
    return filtered_data


def hilbert(data):
    analytic_signals = signal.hilbert(data, axis=-1)
    envelopes = np.abs(analytic_signals)
    phases = np.angle(analytic_signals)
    return analytic_signals, envelopes, phases


def SPLV(data):
    _, _, phases = hilbert(data)
    phase_diffs = phases[:, np.newaxis, ...] - phases[np.newaxis, ...]
    exp_phase_diffs = np.exp(1j * phase_diffs)
    mean_exp_phase_diffs = np.mean(exp_phase_diffs, axis=-1)
    R = np.abs(mean_exp_phase_diffs)
    return R


def Weighted_SPLV(data):
    _, envelopes, phases = hilbert(data)
    phase_diffs = phases[:, np.newaxis, ...] - phases[np.newaxis, ...]
    weights = envelopes[:, np.newaxis, ...] * envelopes[np.newaxis, ...]
    
    # Weighted mean
    weighted_exp_diffs = weights * np.exp(1j * phase_diffs)
    numerator = np.sum(weighted_exp_diffs, axis=-1)
    denominator = np.sum(weights, axis=-1)
    
    R = np.abs(numerator / (denominator + 1e-10))
    return R


def Absolute_Sync_Strength(data):
    """
    Absolute Sync Strength (unnormalized).
    S_ij = |<A_i * A_j * exp(i * delta_phi)>|
    Large only when both amplitudes are large AND their phase difference is stable.
    """
    _, envelopes, phases = hilbert(data)
    phase_diffs = phases[:, np.newaxis, ...] - phases[np.newaxis, ...]
    weights = envelopes[:, np.newaxis, ...] * envelopes[np.newaxis, ...]

    # No normalization: only time-averaging
    sync_vec = weights * np.exp(1j * phase_diffs)
    S = np.abs(np.mean(sync_vec, axis=-1))
    return S


def load_from_json(result_path):
    model = models.from_json(f'{result_path}/env/model_config.json')
    dsg = DatasetGenerator.from_json(f'{result_path}/env/dsg_config.json')
    with open(f'{result_path}/history.json', 'r') as f:
        history = json.load(f)
    return model, dsg, history


#  ------------------------ Plotting helpers ------------------------

def plot_loss(history, yscale="linear"):
    # Plot the training loss curve
    plt.title('training loss')
    plt.plot(history['loss'])
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.yscale(yscale)
    plt.show()
    plt.close()


def plot_weights(model):
    alpha, W_in, W_rec, bias = model.rnn_layer.get_weights()

    # Input-weight visualization
    plt.figure()
    sns.heatmap(W_in[:4],cmap='coolwarm')
    plt.title('input weights')
    plt.xticks([])
    plt.yticks(ticks=range(4), labels=['θ', 'α', 'β', 'γ'])
    plt.show()

    # Recurrent-weight visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(W_rec, cmap='coolwarm')
    plt.title('recurrent weights')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_InOut(inputs, z, settings, time_length=2000):
    # Pull required variables from the settings/results dictionaries
    Fs = settings['Fs']
    task_cnf = settings['task_config']
    wait_length = task_cnf['wait_length']
    switch_num = task_cnf['switch_num']
    rhythms = task_cnf['rhythms']
    color = settings['color']
    
    t_ax, f_ax = calc_axis(settings['data_length'], task_cnf['period_length'], Fs)
    t_split = split_signal(t_ax, wait_length, switch_num)

    # Split the model output by interval and compute the power spectrum and peak frequency
    z_split = split_signal(z, wait_length, switch_num)
    pz_split = power(z_split, Fs)
    max_Hz = f_ax[np.argmax(pz_split, -1)]
    
    fig = plt.figure(figsize=(switch_num*3, 9), tight_layout=True)

    # Input signal
    ax_i = ax_o = fig.add_subplot(3,switch_num,1)
    for i, k in enumerate(rhythms.keys()):
      ax_i.set_title('input')
      ax_i.plot(t_ax, inputs[0,:,i].numpy().T - 1.5*i, label=k, color=color[k])
      ax_i.set_yticks([])

    # Output signal
    ax_o = fig.add_subplot(3,switch_num,2)
    ax_o.set_title('output')
    ax_o.plot(t_ax, z)

    # Per-interval output
    for i in range(switch_num):
      ax_sp = fig.add_subplot(3,switch_num, switch_num+i+1)
      ax_sp.plot(t_split[i][0:time_length], z_split[i][0:time_length] )

    # Per-interval power spectrum
    rhythm_names = list(rhythms.keys())
    for i in range(switch_num):
      ax_pw_sp = fig.add_subplot(3,switch_num, 2*switch_num+i+1)
      # Light shading for all bands, darker shading for the target band
      for j, (rname, (low, high)) in enumerate(rhythms.items()):
        alpha_fill = 0.3 if j == i else 0.08
        ax_pw_sp.axvspan(low, high, alpha=alpha_fill, color=color[rname])
      ax_pw_sp.plot(f_ax, pz_split[i], color='tab:orange')
      ax_pw_sp.set_yscale('log')
      ax_pw_sp.set_xlim(0,100)
      ax_pw_sp.set_xlabel(f"max Hz: {max_Hz[i]:.4}", fontdict={'size':15})

    # Return the figure
    return fig


def plots_all_signal(data):
    n_axes = data.shape[0]
    n_col = int( np.ceil(np.sqrt(n_axes)) )
    y_min, y_max = np.min(data), np.max(data)
    
    fig = plt.figure(figsize=(n_col*1.5, n_col*1.5))
    for i in range(n_axes):
        ax = fig.add_subplot(n_col, n_col, i+1)
        ax.plot(data[i, :])
        ax.set_ylim(y_min, y_max)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return fig


def plot_all_stft(data, fs, nperseg=256):
    n_axes = data.shape[0]
    n_col = int( np.ceil(np.sqrt(n_axes)) )
    
    fig = plt.figure(figsize=(n_col*1.5, n_col*1.5))
    for i in range(n_axes):
        f, t, Zxx = signal.stft(data[i], fs=fs, nperseg=nperseg)
        ax = fig.add_subplot(n_col, n_col, i+1)
        ax.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        ax.set_xlim([t[0], t[-1]])
        ax.set_ylim([0, 50])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return fig
    

def plot_attractor(data, labels, settings, projection='3d', s_dot=0.1, cut=(500, -500)):
    ''' Plot PCA-projected trajectories, one per interval.
    args:
        data: shape=(switch_num, n_components, time_length)
        labels: list of length switch_num
        settings: dictionary containing 'task_config' and 'color'
        projection: '3d' or '2d'
        s_dot: dot size
        cut: tuple (start, end) to cut the time series for plotting
    '''
    switch_num = settings['task_config']['switch_num']
    color = settings['color']
    s, e = cut

    fig = plt.figure()
    if projection == '3d':
        # 3D scatter
        ax = fig.add_subplot(projection='3d')
        for i in range(switch_num):
            ax.scatter(data[i,0,s:e], data[i,1,s:e], data[i,2,s:e],
                        label=labels[i], color=color[labels[i]], s=s_dot)
    else:
        # 2D scatter
        ax = fig.add_subplot()
        for i in range(switch_num):
            ax.scatter(data[i,0,s:e], data[i,1,s:e], 
                        label=labels[i], color=color[labels[i]], s=s_dot)
    plt.legend()
    plt.show()


def plot_amp_scatter(data):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(data[0], data[1], data[2])
    plt.show()


def plot_amp_histgram(data, labels, settings, filtered_data=None):
    color = settings['color']
    switch_num = data.shape[0]

    fig, axes = plt.subplots(1, switch_num, figsize=(switch_num*4, 5), sharey=True, tight_layout=True)
    
    bins = np.linspace(0,2,30)
    for i in range(switch_num):
        ax = axes[i] if switch_num > 1 else axes
        l = labels[i]
        # Unfiltered
        ax.hist(data[i], bins=bins, label='Raw', color='gray', alpha=0.5)
        # Filtered
        if filtered_data is not None:
            # filtered_data[l] gives the amplitudes for the filter 'l'. 
            # We want the i-th row correspond to the i-th mode.
            filt_amp = filtered_data[l][i]
            ax.hist(filt_amp, bins=bins, label=f'{l} filter', color=color[l], alpha=0.7)

        ax.set_title(f"{l} mode")
        ax.set_xlabel('Amplitude')
        ax.legend()

    axes[0].set_ylabel('number of neuron')
    return fig


def plot_filtered_amp_histgram(labels, settings, filtered_data):
    color = settings['color']
    switch_num = len(labels)

    fig, axes = plt.subplots(1, switch_num, figsize=(switch_num*4, 5), sharey=True, tight_layout=True)
    
    bins = np.linspace(0,2,30)
    for i in range(switch_num):
        ax = axes[i] if switch_num > 1 else axes
        l = labels[i]
        
        # Filtered only
        filt_amp = filtered_data[l][i]
        ax.hist(filt_amp, bins=bins, label=f'{l} filter', color=color[l], alpha=0.7)

        ax.set_title(f"{l} mode")
        ax.set_xlabel('Amplitude')
        ax.legend()

    axes[0].set_ylabel('number of neuron')
    return fig


def plot_amp_corr(data, labels):
    amp_df = pd.DataFrame(data.T, columns=labels)

    sns.pairplot(amp_df)
    plt.show()


def plot_SPLV_each_rhythm(data, labels, settings):
    switch_num = settings['task_config']['switch_num']
    fig, ax = plt.subplots(1, 4)
    for i in range(switch_num):
        R = SPLV(data[i])
        ax[i].set_title(f'{labels[i]} mode')
        ax[i].imshow( R , vmin=0, vmax=1)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_weighted_SPLV_4_modes(filtered_r_split, labels):
    """
    filtered_r_split: dict of mode_name -> data array (switch_num, units, time)
    labels: list of mode names corresponding to the switches
    """
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, mode in enumerate(labels):
        if mode in filtered_r_split:
            # slice i for the i-th mode interval
            data = filtered_r_split[mode][i]
            
            R = Weighted_SPLV(data)
            
            ax[i].set_title(f'{mode} mode\n({mode} filter)')
            im = ax[i].imshow(R, vmin=0, vmax=1)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            plt.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
        else:
            ax[i].set_title(f'{mode} (No Data)')
            ax[i].axis('off')

    plt.tight_layout()
    return fig


def plot_absolute_sync_4_modes(filtered_r_split, labels):
    """
    Plot the unnormalized Absolute Sync Strength.
    Bands with small amplitude yield correspondingly small values, which reflects the
    actual sync structure rather than being normalized away.
    """
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    vmax_list = []
    S_list = []

    # First compute all S to determine the color scale
    for i, mode in enumerate(labels):
        if mode in filtered_r_split:
            data = filtered_r_split[mode][i]
            S = Absolute_Sync_Strength(data)
            S_list.append(S)
            vmax_list.append(np.max(S))
        else:
            S_list.append(None)
    
    for i, mode in enumerate(labels):
        if S_list[i] is not None:
            S = S_list[i]
            vmax_i = np.max(S)
            ax[i].set_title(f'{mode} mode\n({mode} filter)\nmax={vmax_i:.3f}')
            im = ax[i].imshow(S, vmin=0, vmax=vmax_i)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            plt.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
        else:
            ax[i].set_title(f'{mode} (No Data)')
            ax[i].axis('off')

    plt.tight_layout()
    return fig


def plot_PLA(results):
    x = results['x']

    # Instantaneous phase
    instantaneous_phase = np.angle(signal.hilbert(x))

    # Phase synchronization index (Kuramoto order parameter)
    sync_index = np.abs(np.mean(np.exp(1j * instantaneous_phase), axis=0))

    # Plot the result
    plt.plot(sync_index)
    plt.title('Phase Synchronization Index Over Time')
    plt.xlabel('Time')
    plt.ylabel('Synchronization Index')
    plt.show()


def plot_STFT(data, settings):
    # Compute the STFT
    f, t, Zxx = signal.stft(data, settings['Fs'], nperseg=256)
    fig, ax = plt.subplots()
    ax.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    ax.set_ylim((0,100))
    ax.set_title('STFT Magnitude')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    fig.colorbar(ax.collections[0], ax=ax, label='Magnitude')
    return fig


def plot_neuron_detail(results, settings, neuron_idx):
    """ Plot the waveform and instantaneous phase of a single neuron. """
    # Pull required variables from the settings/results dictionaries
    labels = results['labels']
    Fs = settings['Fs']
    data_length = settings['data_length']
    period_length = settings['task_config']['period_length']
    wait_length = settings['task_config']['wait_length']
    switch_num = settings['task_config']['switch_num']
    color = settings['color']
    
    x = results['x']
    x_split = split_signal(x, wait_length, switch_num)
    anasig, envelopes, phases = hilbert(x_split)
    phases = results['phases']
    envelopes = results['envelopes']
    

    t_ax, f_ax = calc_axis(data_length, period_length, Fs)
    t_split = split_signal(t_ax, wait_length, switch_num)
    x_split = split_signal(x, wait_length, switch_num)

    fig = plt.figure(figsize=(3*switch_num,8))
    for i in range(switch_num):
      ax1 = fig.add_subplot(3, switch_num, i+1)
      ax1.plot(t_split[i], x_split[neuron_idx, i, :])

      ax2 = fig.add_subplot(3, switch_num, i+1+switch_num)
      ax2.plot(t_split[i], phases[neuron_idx, i, :], c='tab:orange')

      ax3 = fig.add_subplot(3, switch_num, i+1+2*switch_num, polar=True)
      ax3.plot(phases[neuron_idx, i, 100:-100], envelopes[neuron_idx, i, 100:-100], color[labels[i]])
    plt.tight_layout()
    plt.show()


def plot_internal_state(results, settings, time):
    x = results['x']
    x_analytic = signal.hilbert(x, axis=-1)
    x_envelopes = np.abs(x_analytic)
    x_phases = np.angle(x_analytic)

    z = results['z']
    z_analytic = signal.hilbert(z[0,:,0])
    z_envelopes = np.abs(z_analytic)
    z_phases = np.angle(z_analytic)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,polar=True)
    ax.scatter(x_phases[:,time], x_envelopes[:,time], s=10)
    ax.scatter(z_phases[time], z_envelopes[time])
    plt.show()

def internal_state_animation(results, settings, time_steps=None, fps=100, filename='animation.mp4', attention=[]):
    from matplotlib.animation import FuncAnimation

    if time_steps:
        s, e = time_steps
        time_steps = range(s,e)
    else:
        time_steps = range(settings['data_length'])


    x = results['x']
    x_analytic = signal.hilbert(x, axis=-1)
    x_envelopes = np.abs(x_analytic)
    x_phases = np.angle(x_analytic)

    z = results['z']
    z_analytic = signal.hilbert(z[0,:,0])
    z_envelopes = np.abs(z_analytic)
    z_phases = np.angle(z_analytic)

    # Set up the animation
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, polar=True)
    ax.set_ylim((0,2.0))
    scat1 = ax.scatter([], [], s=10)
    scat2 = ax.scatter([], [])
    scat3 = ax.scatter([], [], color='red')

    def init():
        scat1.set_offsets(np.empty((0, 2)))
        scat2.set_offsets(np.empty((0, 2)))
        scat3.set_offsets(np.empty((0, 2)))
        return scat1, scat2, scat3

    def update(t):
        scat1.set_offsets(np.c_[x_phases[:, t], x_envelopes[:, t]])
        scat2.set_offsets([[z_phases[t], z_envelopes[t]]])
        scat3.set_offsets([[x_phases[attention,t], x_envelopes[attention,t]]])
        return scat1, scat2, scat3

    ani = FuncAnimation(fig, update, frames=time_steps, init_func=init, blit=True)

    # Save the animation
    ani.save(filename, writer='ffmpeg', fps=100)


