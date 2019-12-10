import argparse
import logging
from pathlib import Path

import time
import h5py
import hdf5storage as h5s
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

from dataman.lib.report import fig2html
from dataman.lib.util import run_prb

PRECISION = np.dtype(np.single)
N_SAMPLES = 32
N_CHANNELS = 4
sampling_rate = 3e4

logger = logging.getLogger(__name__)

# Available features
# TODO: per feature CLI arguments
# TODO: feature discovery as modules
AVAILABLE_FEATURES = ['energy', 'cpca', 'chwpca', 'position']


def scale_feature(fet):
    nfet = fet - np.mean(fet, axis=0)
    nfet /= np.std(nfet, axis=0)
    return nfet


def feature_peak(wv, minimum=True, absolute=False):
    """Calculate per-channel peak amplitude."""
    if minimum:
        return wv.max(axis=0).T

    if absolute:
        return wv.abs().max(axis=0).T

    if absolute and minimum:
        raise NotImplementedError('Minimum and absolute do not work together.')


def feature_energy(wv):
    """Calculate l2 (euclidean) norm for vector containing spike waveforms
    """
    return np.sqrt(np.sum(wv ** 2, axis=0)).T


def feature_position(pos_file, dat_offsets, timestamps, indices, sampling_rate=3e4):
    raise NotImplemented('This feature has not been implemented yet.')
    pos_f = h5py.File(pos_file, 'r')
    positions = np.array(pos_f['XY_data']).T
    av_pos = (np.nanmax(positions) - np.nanmin(positions)) / 2

    n_frames = len(positions)
    n_records = [(nb - 1024) / 2070 for nb in n_bytes]
    assert not any([nr % 1 for nr in n_records])
    n_samples = [int(nr * 1024) for nr in n_records]

    starts = [sum(n_samples[:n]) for n in range(len(n_samples))]
    s_duration = [ns / sampling_rate for ns in n_samples]
    fps = n_frames / s_duration[1]

    idx_vid_start = np.nonzero(indices > starts[1])[0].min()
    idx_vid_end = np.nonzero(indices > starts[2])[0].min() - 1

    t_pos = np.zeros(len(timestamps))
    t_vel = np.zeros_like(t_pos)
    t_acc = np.zeros_like(t_pos)

    t_pos[np.squeeze(timestamps) < starts[1] / 3] = -1
    t_pos[np.squeeze(timestamps) > starts[2] / 3] = -1

    for idx in range(idx_vid_start, idx_vid_end):
        vid_index = int((indices[idx] - n_samples[0]) * fps // sampling_rate)

        t_pos[idx] = positions[vid_index]

    # fill nans by interpolation

    t_vel[1:] = np.diff(t_pos)
    t_acc[1:] = np.diff(t_vel)

    return np.vstack([t_pos, t_pos, t_vel, t_acc]).T


def feature_cPCA(wv, n_components=12, incremental=False, batch_size=None):
    if incremental:
        raise NotImplementedError("Can't run incremental PCA yet.")

    ers = np.reshape(np.transpose(wv, axes=(1, 0, 2)), (N_SAMPLES * N_CHANNELS, -1))
    pca = PCA(n_components)
    scores = pca.fit_transform(ers.T)
    return scores


def feature_chwPCA(wv, dims=3, energy_normalize=False):
    """
    waveforms shape is (nSamples x nChannels x nSpikes)
    """
    pcas = []
    pca_scores = []
    for d in range(4):
        pca = PCA(n_components=dims)
        data = wv[:, d, :].T.copy()
        data -= np.mean(data, axis=0)
        if energy_normalize:
            l2 = feature_energy(wv)
            data = data / np.expand_dims(l2[:, d], axis=1)
        pca_scores.append(pca.fit_transform(data))
        pcas.append(pca)
    pca_scores = np.concatenate(pca_scores, axis=1)
    return pca_scores


def plot_waveforms_grid(wv, n_rows=10, n_cols=20, n_overlay=10000):
    n_overlay = min(n_overlay, wv.shape[2])

    target_wv = wv[:, :, np.linspace(0, wv.shape[2] - 1, n_rows * n_cols, dtype='int64')]
    max_amplitude = target_wv.max(axis=(0, 1, 2))
    min_amplitude = target_wv.min(axis=(0, 1, 2))

    fig = plt.figure(figsize=(28, 10))

    # gridspec inside gridspec
    outer_grid = gridspec.GridSpec(1, 2, wspace=0.0, hspace=0.0, width_ratios=[20, 8])
    inner_grid = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=outer_grid[0], wspace=0.0, hspace=0.0)

    # waveform plots
    for nr in range(n_rows):
        for nc in range(n_cols):
            n = nc + nr * n_cols
            ax = plt.Subplot(fig, inner_grid[n])
            ax.axis('off')
            ax.plot(target_wv[:, :, n], linewidth=1)
            ax.set_ylim(min_amplitude, max_amplitude)
            fig.add_subplot(ax)

    target_wv = wv[:, :, np.linspace(0, wv.shape[2] - 1, n_overlay, dtype='int64')]
    ch_grid = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_grid[1], wspace=0.0, hspace=0.0)
    for nr in range(2):
        for nc in range(2):
            n = nc + nr * 2
            ax = plt.Subplot(fig, ch_grid[n])
            ax.plot(target_wv[:, n, :], linewidth=1, c=f'C{n}', alpha=.02)
            ax.axis('off')
            ax.set_ylim(min_amplitude, max_amplitude)
            fig.add_subplot(ax)

    return fig


def plot_feature(feature, feature_name='', timestamps=None):
    num_features = feature.shape[1]
    if num_features > 16:
        raise ValueError("More than 16 features found, indicating wrong array shape.")

    fig, ax = plt.subplots(2, num_features, figsize=(16, 6))
    yq_l, yq_h = np.quantile(feature[:, 0], .1), np.quantile(feature[:, 0], .9)

    for n, num_fet in enumerate(range(0, num_features)):
        lim_l, lim_h = np.quantile(feature[:, num_fet], .1), np.quantile(feature[:, num_fet], .9)
        if n:
            ax[0, n].scatter(feature[:, 0], feature[:, num_fet], s=.2, alpha=.1)
            ax[0, n].set_title('{fn}:{n} / {fn}:0'.format(fn=feature_name, n=num_fet))
            ax[0, n].set_xlim(yq_l, yq_h)
            ax[0, n].set_ylim(lim_l, lim_h)
        else:
            # TODO: PLOT EVENTS PER SECOND
            ax[0, n].plot(timestamps)

        ax[1, n].scatter(timestamps, feature[:, num_fet], s=.2, alpha=.1)
        ax[1, n].set_ylim(lim_l, lim_h)

    return fig


def write_features_fet(feature_data, outpath):
    start = time.time()
    # TODO: Channel validity
    fet_data = np.hstack(list(feature_data))
    with open(outpath, 'w') as fet_file:
        fet_file.write(f' {fet_data.shape[1]}\n')

        np.savetxt(fet_file, fet_data, fmt='%-16.8f', delimiter=' ')
        # for row in fet_data:
        #     fet_file.writelines(' '.join(map(str, list(row))))
        #     fet_file.write('\n')
    print(time.time() - start)


def write_feature_fd(feature_names, feature_data, timestamps, outpath, tetrode_path, channel_validity=None):
    if type(feature_names) == str:
        feature_names = [feature_names]

    feature_name_ch = ['{}: {}'.format(fn, n + 1) for fn in feature_names for n in range(feature_data.shape[1])]
    feature_std = feature_data.std(axis=0, dtype='double')
    feature_av = feature_data.mean(axis=0, dtype='double')

    if channel_validity is None:
        channel_validity = [1, 1, 1, 1]

    if len(feature_names) > 1:
        raise NotImplementedError('No logic for multiple features yet. External or just .fd?')

    outpath_fname = outpath / '{}_{}.fd'.format(tetrode_path.stem, feature_names[0])

    h5s.savemat(outpath_fname, {'ChannelValidity': np.array(channel_validity, dtype='double'),  # dead channel index
                                'FD_av': feature_av,  # mean
                                'FD_sd': feature_std,  # sd
                                'FeatureData': feature_data,
                                'FeatureIndex': np.arange(1, feature_data.shape[0] + 1, dtype='double'),
                                'FeatureNames': feature_name_ch,
                                'FeaturePar': [],
                                'FeaturesToUse': feature_names,
                                'FeatureTimestamps': timestamps,
                                'TT_file_name': str(tetrode_path.name),
                                }, compress=False, truncate_existing=True, truncate_invalid_matlab=True,
                appendmat=False)


def main(args):
    parser = argparse.ArgumentParser('Generate .fet and .fd files for features from spike waveforms')
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose (debug) output")

    parser.add_argument('target', default='.', help="""Directory with waveform .mat files.""")
    parser.add_argument('-o', '--out_path', help='Output file path Defaults to current working directory')
    parser.add_argument('--sampling-rate', type=float, help='Sampling rate. Default 30000 Hz', default=3e4)
    parser.add_argument('-f', '--force', action='store_true', help='Force overwrite of existing files.')
    parser.add_argument('-a', '--align', help='Alignment method, default: min', default='min')
    parser.add_argument('-F', '--features', nargs=-1, help='Features to use. Default: energy', default=['energy'])
    parser.add_argument('--ignore-prb', action='store_true',
                        help='Do not load channel validity from dead channels in .prb files')
    parser.add_argument('--no-report', action='store_true', help='Do not generate report file (saves time)')
    cli_args = parser.parse_args(args)

    matpath = Path(cli_args.target).resolve()
    matfiles = sorted(list(map(Path.resolve, matpath.glob('tetrode??.mat'))))
    logger.debug([mf.name for mf in matfiles])
    logger.info('Found {} waveform files'.format(len(matfiles)))

    # TODO:
    # per feature arguments

    for nt, matfile in tqdm(enumerate(matfiles), total=len(matfiles)):
        outpath = matfile.parent / 'FD'
        if not outpath.exists():
            outpath.mkdir()

        # Load prb file if it exists and set channel validity based on dead channels
        prb_path = matfile.with_suffix('.prb')
        if prb_path.exists():
            prb = run_prb(prb_path)
        else:
            logging.warning(f'No probe file found for {matfile} and no channel validity given.')
            prb = None
        if prb is None or 'dead_channels' not in prb:
            channel_validity = [1, 1, 1, 1]
        else:
            channel_validity = [int(ch not in prb['dead_channels']) for ch in prb['channel_groups'][0]['channels']]
        logging.debug('channel validity: {}'.format(channel_validity) + ('' if all(
            channel_validity) else f', {4 - sum(channel_validity)} dead channel(s)'))

        hf = h5py.File(matfile, 'r')
        waveforms = np.array(hf['spikes'], dtype=PRECISION).reshape(N_SAMPLES, N_CHANNELS, -1)

        timestamps = np.array(hf['index'], dtype='double')
        # indices = timestamps * sampling_rate / 1e4

        features = {}
        for fet_name in map(str.lower, cli_args.features):
            if fet_name == 'energy':
                logging.debug(f'Calculating {fet_name} feature')
                features['energy'] = scale_feature(feature_energy(waveforms))

            elif fet_name == 'peak':
                logging.debug(f'Calculating {fet_name} feature')
                features['peak'] = feature_peak(waveforms)

            elif fet_name == 'cpca':
                logging.debug(f'Calculating {fet_name} feature')
                cpca = scale_feature(feature_cPCA(waveforms))
                print('cpca shape', cpca.shape)
                features['cPCA'] = cpca

            elif fet_name == 'chwpca':
                logging.debug(f'Calculating {fet_name} feature')
                chwpca = scale_feature(feature_chwPCA(waveforms))
                print('chw pca', chwpca.shape)
                features['chwPCA'] = chwpca

        # TODO:
        # fet_cpca_4 = fet_cpca[:, :4]

        # # Position feature
        # n_bytes = [250154314, 101099824, 237970294]
        # fet_pos = feature_position(matpath / 'XY_data.mat', dat_offsets=n_bytes, timestamps=timestamps,
        #                            indices=indices)

        with open(matfile.with_suffix('.html'), 'w') as fet_report_file:
            logging.debug('Generating waveform graphic')
            fig = plot_waveforms_grid(waveforms)
            fet_report_file.write(fig2html(fig))
            del fig

        fet_file_path = outpath / matfile.with_suffix('.fet.0').name
        logging.debug(f'Writing .fet file {fet_file_path}')
        write_features_fet(feature_data=features.values(), outpath=fet_file_path)

        # feature_names = ['energy', 'cPCA']
        # fd_features = [fet_energy, fet_cpca_4]
        for fet_name, fet_data in features.items():
            write_feature_fd(feature_names=fet_name, feature_data=fet_data,
                             timestamps=timestamps, outpath=outpath, tetrode_path=matfile,
                             channel_validity=channel_validity)
