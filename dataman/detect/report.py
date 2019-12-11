import io
import urllib
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import datashader as ds

plt.rcParams['figure.figsize'] = 15, 8

DS_CMAPS = tuple([plt.cm.get_cmap(c) for c in ['Blues', 'Oranges', 'Greens', 'Reds']])


def fig2html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)

    string = base64.b64encode(buf.read())

    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "{}"/>'.format(uri)
    return html


def plot_noise(noise_arr, thresholds, tetrode=None):
    fig, ax = plt.subplots(4, 1, figsize=(18, 4), sharex=True)
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


def ds_shade_to_html(shade):
    img_string = base64.b64encode(shade.to_bytesio(format='png').read())

    uri = 'data:image/png;base64,' + urllib.parse.quote(img_string)
    html = '<img src = "{}"/>'.format(uri)
    return html


def plot_shades(shades, how, y_min_uv=-500, y_max_uv=400):
    fig, axes = plt.subplots(1, len(shades), figsize=(16, 6))
    fig.suptitle('All waveforms, density "{}" scaled.'.format(how))
    for n, ax in enumerate(axes):
        # cast rastered shade into png image as workaround to false color application by imshow??
        # TODO: That can't be the best way to go about this!!
        img_arr = mpimg.imread(shades[n].to_bytesio('png'))
        h, w = img_arr.shape[:2]

        ax.imshow(img_arr)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: '{:.0f}'.format(x / w * 32)))
        if not n:
            # replace axis ticks of image with actual amplitude scale in microvolts
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, p: '{:.0f}'.format(((-y / h * (y_max_uv - y_min_uv)) + y_max_uv))))
        else:
            ax.set_yticklabels([])
    plt.tight_layout()
    return fig


def ds_shade_waveforms(waveforms, canvas_width=400, canvas_height=400, color_maps=None, y_min_uv=-500, y_max_uv=400,
                       how='log'):
    n_samples = waveforms.shape[0]
    n_channels = waveforms.shape[1]

    if color_maps is None:
        color_maps = DS_CMAPS

    if n_channels > len(color_maps):
        color_maps *= len(color_maps) // n_channels + 1

    cvs = ds.Canvas(plot_height=canvas_height, plot_width=canvas_width,
                    x_range=(0, waveforms.shape[0] - 1),
                    y_range=(y_min_uv / 0.195, y_max_uv / 0.195))

    shades = []
    for ch in range(n_channels):
        df = pd.DataFrame(waveforms[:, ch, :].T)
        agg = cvs.line(df, x=np.arange(n_samples), y=list(range(n_samples)), agg=ds.count(), axis=1)
        img = ds.transfer_functions.shade(agg, how=how, cmap=plt.cm.get_cmap(color_maps[ch]))
        shades.append(img)

    # If we wanted to use all channels together, we have to create a categorical column for the selection
    # df = pd.DataFrame(wv_reshaped)
    # df['channel'] = pd.Categorical(np.tile(np.arange(n_channels), wv_reshaped.shape[0]//n_channels))
    # wv_reshaped = waveforms.reshape(n_samples, -1).T
    # agg = cvs.line(df, x=np.arange(n_samples), y=list(range(32)), agg=ds.count_cat('channel'), axis=1)

    return shades
