import io
import urllib
import base64
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 15, 8


def fig2html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    string = base64.b64encode(buf.read())

    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "{}"/>'.format(uri)
    return html


def plot_noise(noise_arr, thresholds, tetrode=None):
    fig, ax = plt.subplots(4, 1, figsize=(18, 6), sharex=True)
    #     fig = plt.figure()
    #     ax = []
    #     ax[0] = plt.subplot2grid((8, 4), (0, 0))
    #     ax2 = plt.subplot2grid((8, 4), (0, 1), colspan=2)
    #     ax3 = plt.subplot2grid((8, 4), (1, 0), colspan=2, rowspan=2)
    #     ax4 = plt.subplot2grid((8, 4), (1, 2), rowspan=2)

    title = 'Noise estimation (1.0 second bins) ' + ('' if tetrode is None else f'tetrode {tetrode}')
    ax[0].set_title(title)

    t = np.linspace(0, len(noise_arr), len(noise_arr))
    limits = np.min(np.percentile(noise_arr, 0, axis=0)), np.max(np.percentile(noise_arr, 99, axis=0))

    for n in range(4):
        ax[n].plot(t, noise_arr[:, n], color='C' + str(n))
        ax[n].set_ylabel('$\mu V$')
        ax[n].set_ylim(limits)

        # draw thresholds
        ax[n].axhline(thresholds[n], linestyle=':', color='gray')

    ax[-1].set_xlabel('$Seconds$')
    return fig


def plot_raw(arr, fs=3e4):
    amplitude_correction = 0.195
    fig, ax = plt.subplots(figsize=(18, 4))

    # Limit offset and width in seconds of data we are going to plot
    dur = int(.5 * fs)
    ofs = int(0 * fs)
    t = np.linspace(ofs, ofs + dur, dur) / fs

    # Plot, but add a fixed value to each channel to separate them for display
    ax.plot(t, arr[ofs:ofs + dur, :] * amplitude_correction + np.linspace(0, 4 * 200, 4))
    ax.set_xlabel('$Seconds$')
    ax.set_ylabel('$\mu V$')

    return fig
