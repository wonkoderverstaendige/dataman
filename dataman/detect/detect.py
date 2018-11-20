import numpy as np
import argparse
import logging
from scipy import signal, interpolate
from tqdm import tqdm, trange
from pathlib import Path
from datetime import datetime as dt
from dataman.detect import report
import matplotlib.pyplot as plt

from scipy.io import savemat

logger = logging.getLogger(__name__)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def get_batches(length, batch_size):
    batches = [bc * batch_size for bc in range(length // batch_size)]
    if length - (length // batch_size) * batch_size:
        batches.append(length - length % batch_size)
    return batches


def estimate_noise(array, lc=300, hc=6000, num_channels=4, fs=3e4, uV_factor=0.195):
    ne_binsize = int(fs)  # noise estimation binsize

    # Filter
    b, a = butter_bandpass(lc, hc, fs)
    batches = get_batches(array.shape[0], ne_binsize)
    ne = np.zeros((len(batches), num_channels))
    nfac = 1 / 0.6745

    for n, batch in enumerate(tqdm(batches, leave=False, desc='estimation: ')):
        l = min(ne_binsize, array.shape[0] - batch)
        filtered = signal.filtfilt(b, a, array[batch:batch + l, :].astype(np.double), axis=0) * uV_factor
        for ch in range(num_channels):
            ne[n, ch] = np.median(abs(filtered[:, ch]) * nfac)
    return ne


def detect_spikes(arr, thresholds, num_channels=4, fs=3e4, uV_factor=0.195, chunk_size_s=60, lc=300, hc=6000):
    chunk_size = int(chunk_size_s * fs)  # chunk size for detection

    use_thr = thresholds
    max_thr = use_thr * 8

    waveform_chunks = []
    timestamp_chunks = []

    spikes_per_chunk = []
    rejections = 0

    INTERP_F = 4
    b, a = butter_bandpass(lc, hc, fs)

    # add padding samples for later interpolation
    s_pad = 0
    n_samples = 32 + 2 * s_pad

    end = arr.shape[0]

    batches = [b * chunk_size for b in range(end // chunk_size)]
    if arr.shape[0] - (arr.shape[0] // chunk_size) * chunk_size:
        batches.append(arr.shape[0] - arr.shape[0] % chunk_size)

    for batch in tqdm(batches, leave=False, desc='extraction: '):
        start = batch
        end = start + min(chunk_size, arr.shape[0] - start)

        # Filtering and Detection
        filtered = signal.filtfilt(b, a, arr[start:end, :].astype(np.double), axis=0) * uV_factor

        # Merge threshold crossings
        # TODO: Only merge valid channels!
        crossings = filtered < -use_thr
        crossings = np.sum(filtered < -use_thr, axis=1)
        crossings = np.clip(crossings, 0, 1)

        # make sure crossings in the first 30 samples are ignored to avoid wrong start
        crossings[:30] = 0

        # Look for threshold crossing "onset"
        deltas = np.diff(crossings)
        xr_starts = (deltas > 0).nonzero()[0]
        xr_ends = (deltas < 0).nonzero()[0]

        assert len(xr_starts) == len(xr_ends)
        spike_idx = (np.array([xr_starts, xr_ends]).T.mean(axis=1)).astype(np.int64)
        #     print(f'Found {spike_idx.shape[0]} spikes!')

        # Check if last spike is too close to boundary
        while len(spike_idx) and spike_idx[-1] + 22 >= chunk_size:
            spike_idx = spike_idx[:-1]
            rejections += 1

        spikes_per_chunk.append(len(spike_idx))

        if not len(spike_idx):
            continue

        # Extract waveforms
        n_spikes = spike_idx.shape[0]
        bc_samples = np.arange(-10 - s_pad, 22 + s_pad).reshape([1, n_samples])
        bc_spikes = spike_idx.reshape([n_spikes, 1])

        idc = bc_samples + bc_spikes

        try:
            wv = filtered[idc]
        except IndexError:
            print(spike_idx[-1] + 22)
            break

        # Temporary arrays
        spk_wv = np.zeros((n_spikes, 32, 4), dtype=np.int32)
        spk_ts = np.zeros(n_spikes, dtype=np.int64)
        min_idx = np.zeros(n_spikes, dtype=np.int16)

        x = np.linspace(0, n_samples, n_samples)
        wv_f_interp = interpolate.interp1d(x, wv, axis=1, kind='cubic')

        x_interp = np.linspace(0, n_samples, n_samples * INTERP_F)
        wv_interp = wv_f_interp(x_interp)

        reject = np.zeros(len(spk_ts))
        for n in range(len(spk_ts)):  # tnrange(n_spikes)
            # interpolated waveform
            w = wv_interp[n]

            # interpolated minimum
            mins = np.min(w, axis=0)

            # index of minimum of channel with largest amplitude valley
            ch_sel = np.argmin(mins)
            idx = np.argmin(w[:, ch_sel])
            min_idx[n] = idx

            if np.any(mins < -max_thr):
                reject[n] = True
                rejections += 1

        spk_ts = spike_idx - 10 + np.round(min_idx / INTERP_F).astype(np.int64)
        spk_ts = spk_ts[np.logical_not(reject)]

        # Check again if last spike is too close to boundary
        while len(spk_ts) and spk_ts[-1] + 22 >= chunk_size:
            spk_ts = spk_ts[:-1]
            rejections += 1

        bc_samples = np.arange(-10 - s_pad, 22 + s_pad).reshape([1, n_samples])
        bc_spikes = spk_ts.reshape([spk_ts.shape[0], 1])

        idc = bc_samples + bc_spikes

        spk_wv = filtered[idc]

        waveform_chunks.append(spk_wv)
        timestamp_chunks.append(spk_ts + int(start))

    waveforms = np.vstack(waveform_chunks)
    timestamps = np.concatenate(timestamp_chunks) / fs * 10000  # convert to sampling rate and MClust time format

    # Reorder timestamps and waveforms
    ts_order = np.argsort(timestamps)
    timestamps.sort()
    waveforms = waveforms[ts_order]

    return waveforms, timestamps


def main(args):
    parser = argparse.ArgumentParser('Detect spikes in .dat files')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="Verbose (debug) output")

    parser.add_argument('target', default='.',
                        help="""Directory with tetrode files.""")
    parser.add_argument('-o', '--out_path', help='Output file path Defaults to current working directory')
    parser.add_argument('--sampling-rate', type=float, help='Sampling rate. Default 30000 Hz', default=3e4)
    parser.add_argument('-p', '--prefix', help='Input filename template, defaults to "tetrode{:02d}.dat"',
                        default='tetrode{:02d}.dat')
    parser.add_argument('-m', '--matfile', help="""Filename for output. For easy further processing, defaults to
                                                  'tetrode{:02d}.mat'""", default='tetrode{:02d}.mat')
    parser.add_argument('-t', '--tetrodes', nargs='*', help='0-index list of tetrodes to look at. Default: all.')
    parser.add_argument('-f', '--force', action='store_true', help='Force overwrite of existing files.')
    parser.add_argument('--start', type=float, help='Segment start in seconds')
    parser.add_argument('--end', type=float, help='Segment end in seconds')

    cli_args = parser.parse_args(args)
    logger.debug('Arguments: {}'.format(cli_args))

    report_string = ''
    fs = cli_args.sampling_rate
    noise_percentile = 5
    target = Path(cli_args.target)
    tetrodes = range(16) if (cli_args.tetrodes is None or not len(cli_args.tetrodes)) else map(int, cli_args.tetrodes)

    start = int(cli_args.start * fs)
    end = int(cli_args.end * fs)

    for tetrode in tqdm(list(tetrodes), desc='Progress'):
        tqdm.write(f'Detecting spikes for tetrode {tetrode}')
        path = (target / cli_args.prefix.format(tetrode)).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        matpath = path.parent / cli_args.matfile.format(tetrode)
        if matpath.exists() and not cli_args.force:
            print(f'{matpath} already exists. Delete or use --force to overwrite.')
            raise SystemExit

        if None not in (start, end):
            assert start < end
            wb = np.memmap(path, dtype='int16').reshape((-1, 4))[start:end]
        else:
            wb = np.memmap(path, dtype='int16').reshape((-1, 4))

        report_string += '<h1>Recording</h1>\n'
        report_string += 'Length: {:.1f} Msamples, {:.1f} minutes'.format(wb.shape[0] / 1e6, wb.shape[0] / fs / 60)

        report_string += f'<h1>Tetrode {tetrode}</h1>'
        report_string += str(path) + '<br>'

        fig = report.plot_raw(wb)
        report_string += report.fig2html(fig) + '<br>'
        plt.close(fig)
        del fig

        noise = estimate_noise(wb)
        ne05 = np.percentile(noise, noise_percentile, axis=0)
        ne_min = np.min(noise, axis=0)
        ne_max = np.max(noise, axis=0)
        ne_std = np.std(noise, axis=0)

        # Report noise amplitudes
        report_string += '<h2>Noise estimation</h2>'
        fig = report.plot_noise(noise, thresholds=ne05, tetrode=tetrode)
        report_string += report.fig2html(fig) + '<br>'
        plt.close(fig)
        del fig

        thr = ne05 * 4.5
        for ch in range(4):
            l = '<b>Channel {}:</b>, Threshold: {:.1f}, 5th percentile: {:.1f} uV (min: {:.1f}, max: {:.1f}, std: {:.1f})'
            report_string += l.format(ch, thr[ch], ne05[ch], ne_min[ch], ne_max[ch], ne_std[ch]) + '<br>'

        # Spike Detection
            report_string += '<h2>Spike Detection</h2>'

        waveforms, timestamps = detect_spikes(wb, thr)

        report_string += '<b>{} spikes</b> ({:.1f} sps)'.format(len(timestamps),
                                                         len(timestamps) / (wb.shape[0] / 3e4))

        # Write to file
        mdict = {'spikes': waveforms.reshape(-1, 128).astype(np.double),
                 'index': timestamps.reshape(-1, 1).astype(np.double), 'n': len(timestamps)}

        savemat(str(matpath), mdict=mdict, format='5', do_compression=False)
        report_string += '<hr>'

    now = dt.today().strftime('%Y%m%d_%H%M%S')
    with open(path.parent / f'pydetect_report_{now}.html', 'w') as rf:
        rf.write(report_string)

    print('Done!')
