import sys
import argparse
import logging
import shutil
import subprocess
from itertools import combinations
from pathlib import Path

import datashader as ds  # slow to import, takes ~4 s
import hdf5storage as h5s
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pkg_resources
import yaml
from datashader import transfer_functions as ds_tf

from dataman.lib.report import fig2html, ds_plot_features

logger = logging.getLogger(__name__)


def load_yaml(yaml_path):
    """Load content of a .yaml file as dictionary
    """
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError('Invalid YAML path {}'.format(yaml_path))

    with open(yaml_path, 'r') as yf:
        return yaml.load(yf, Loader=yaml.SafeLoader) or {}


def run_kk(params, run_kk=True):
    cfg, target_path = params
    tt_fname = target_path.name
    tetrode_file_stem = tt_fname.split(".")[0]
    tetrode_file_elecno = tt_fname.split(".")[-1]
    working_dir = target_path.parent
    logging.debug(f'Tetrode name: {tt_fname}, stem: {tetrode_file_stem}, ElecNo: {tetrode_file_elecno}')
    clu_file = working_dir / (tetrode_file_stem + f'.clu.{tetrode_file_elecno}')
    if clu_file.exists() and cfg['skip']:
        logging.error(f'Clu file {clu_file} exists. Skipping.')
        run_kk = False

    # Read in feature validity
    validity_path = target_path.with_suffix('.validity')
    if not validity_path.exists():
        logger.warning('No explicit feature validity given, falling back to default = all used.')
    with open(validity_path) as vfp:
        validity_string = vfp.readline()
    logger.debug(f'Channel validity: {validity_string}')

    # Combine executable and arguments
    kk_executable = cfg["kk_executable"]
    kk_cmd = f'{kk_executable} {tetrode_file_stem} -ElecNo {tetrode_file_elecno} -UseFeatures {validity_string}'
    if cfg['KKv3']:
        kk_cmd += ' -UseDistributional 0'

    # additional command line options
    if (cfg.kk_additional_args):
        kk_cmd += ' ' + cfg.kk_additional_args

    kk_cmd_list = kk_cmd.split(' ')
    logger.debug(f'KK COMMAND: {kk_cmd}')
    logger.debug(f'KK COMMAND LIST: {kk_cmd_list}')

    # Call KlustaKwik and gather output
    # TODO: Use communicate to interact with KK, i.e. write to log and monitor progress
    #       see https://stackoverflow.com/questions/21953835/run-subprocess-and-print-output-to-logging
    logger.info('Starting KlustaKwik process')
    if cfg['PRINT_KK_OUTPUT']:
        stdout = None
    else:
        stdout = subprocess.PIPE

    if run_kk:
        kk_call = subprocess.run(kk_cmd_list, stderr=subprocess.STDOUT, stdout=stdout)
        kk_error = kk_call.returncode

        logger.debug('Writing KlustaKwik log file')
        logger.debug('Clu File: ' + str(clu_file))
        if kk_call.stdout is not None:
            with open(clu_file.with_suffix('.log'), 'w') as log_file:
                log_file.write(kk_call.stdout.decode('ascii'))
        else:
            logging.warning('Missing stdout, not writing log file!')

        # Check call return code and output
        if kk_error:
            logging.error(f'KlustaKwik error code: {kk_error}')
            exit(kk_error)
        else:
            logging.debug(f'KlustaKwik successful: {kk_error}')

    # Load clu file
    logger.debug(f'Loading {clu_file}')
    clu_df = pd.read_csv(clu_file, dtype='category', names=['cluster_id'], skiprows=1)
    cluster_labels = clu_df['cluster_id'].cat.categories
    num_clusters = len(cluster_labels)
    logger.info(f'{len(clu_df)} spikes in {num_clusters} clusters')

    # Find all feature .fd files
    feature_files = list(working_dir.glob(tetrode_file_stem + '_*.fd'))
    ff_sizes = [ff.stat().st_mtime for ff in feature_files]
    feature_files = [f for t, f in sorted(zip(ff_sizes, feature_files))]
    if not len(feature_files):
        raise FileNotFoundError(f'No Feature Files found in {working_dir}')

    # TODO: Stupid, the feature names are in the .fd file already
    feature_names = [str(ff.name).split(tetrode_file_stem + '_')[1].split('.')[0] for ff in feature_files]
    logger.info(f'Loading features: {feature_names}')

    color_keys = cfg['CLUSTER_COLORS']
    with open(clu_file.with_suffix('.html'), 'w') as crf:
        crf.write('<head></head><body><h1>{}</h1>'.format(clu_file.name))
        for fd_file, fet_name in zip(feature_files, feature_names):
            crf.write('<h3>Feature: {}</h3>\n'.format(fet_name))
            logger.info(f'Generating images for feature {fet_name}')
            if not fd_file.exists():
                continue

            logger.debug(f'Loading {fd_file}')
            mat_fet = h5s.loadmat(str(fd_file), appendmat=False)

            fd_df = pd.DataFrame(mat_fet['FeatureData'])
            fd_df.rename(columns={c: str(c) for c in fd_df.columns}, inplace=True)
            if not len(clu_df) == len(fd_df):
                raise ValueError(f'Number of cluster labels ({num_clusters}) does not match number of spikes'
                                 f'in {fd_file} ({len(fd_df)})')

            fd_df['clu_id'] = clu_df.cluster_id.astype('category')
            logger.debug(f'Feature {fet_name} loaded with {len(fd_df)} spikes, {fd_df.shape[1] - 1} dimensions ')

            images = []
            titles = []
            for cc in combinations(map(str, range(len(fd_df.columns) - 1)), r=2):
                fet_title = f'x: {fet_name}:{cc[0]} vs y: {fet_name}:{cc[1]}'
                x_range = (np.percentile(fd_df[cc[0]], 0.01), np.percentile(fd_df[cc[0]], 99.9))
                y_range = (np.percentile(fd_df[cc[1]], 0.01), np.percentile(fd_df[cc[1]], 99.9))

                logger.debug(f'shading {len(fd_df)} points in {fd_df.shape[1] - 1} dimensions')
                canvas = ds.Canvas(plot_width=300, plot_height=300, x_range=x_range, y_range=y_range)
                try:
                    agg = canvas.points(fd_df, x=cc[0], y=cc[1], agg=ds.count_cat('clu_id'))
                    with np.errstate(invalid='ignore'):
                        img = ds_tf.shade(agg, how='log', color_key=color_keys)
                        img = img if cfg['no_spread'] else ds_tf.spread(img, px=1)
                except ZeroDivisionError:
                    img = None
                images.append(img)
                titles.append(fet_title)

            logger.debug(f'Creating plot for {fet_name}')
            fet_fig = ds_plot_features(images, how='log', fet_titles=titles)
            crf.write(fig2html(fet_fig) + '</br>\n')
            plt.close(fet_fig)


def main(args):
    parser = argparse.ArgumentParser('Clustering with KlustaKwik')
    parser.add_argument('target', help='Target path, either path containing tetrode files, or single tetrodeXX.mat')
    parser.add_argument('--KK', help='Path to KlustaKwik executable')
    parser.add_argument('--features', nargs='*', help='list of features to use for clustering')
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--skip', help='Skip if clu file exists already', action='store_true')
    parser.add_argument('--no_spread', help='Shade report plots without static spread', action='store_true')
    parser.add_argument('--kkargs', help='Additional KK parameters, default: {-MaxPossibleClusters 35}',
                        type=str, default='-MaxPossibleClusters 35')
    parser.add_argument('-N', '--num_proc',
                        help='Number of KlustaKwik instances to run in parallel, defaults to 0 (all)', type=int,
                        default=0)
    parser.add_argument('--KKv3', action='store_true',
                        help='Running KlustaKwik v3 requires additional parameters for the call.')
    cli_args = parser.parse_args(args)

    # Load default configuration yaml file
    default_cfg_path = Path(pkg_resources.resource_filename(__name__, '../resources/cluster_defaults.yml')).resolve()
    if not default_cfg_path.exists():
        logging.error('Could not find default config file.')
        raise FileNotFoundError
    logger.debug('Loading default configuration')
    cfg = load_yaml(default_cfg_path)

    # Load local config file if it exists
    local_cfg_path = Path(
        pkg_resources.resource_filename(__name__, '../resources/cluster_defaults_local.yml')).resolve()
    if local_cfg_path.exists():
        logger.debug('Loading and updating with local configuration')
        local_cfg = load_yaml(local_cfg_path)
        cfg.update(local_cfg)

    # Load custom config path
    custom_cfg_path = Path(cli_args.config).resolve() if cli_args.config else None
    if custom_cfg_path:
        if custom_cfg_path.exists():
            logger.debug('Loading and updating with custom configuration')
            cfg.update(load_yaml(custom_cfg_path))
        else:
            raise FileNotFoundError(f"Could not load configuration file {custom_cfg_path}")

    # Load parameters from command line
    logger.debug('Parsing and updating configuration with CLI arguments')
    cfg.update(vars(cli_args))

    # try to find KlustaKwik executable if necessary...
    if cli_args.KK is None:
        cli_args.KK = shutil.which('KlustaKwik') or shutil.which('klustakwik') or shutil.which('Klustakwik')
    if cli_args.KK is None:
        raise FileNotFoundError('Could not find the KlustaKwik executable on the path, and none given.')

    cfg['kk_executable'] = cli_args.KK
    cfg['kk_additional_args'] = cli_args.kkargs

    logger.debug(cfg)

    # 1) Find target file stem
    target_path = Path(cli_args.target).resolve()
    if target_path.is_file():
        tetrode_files = [target_path]
        logger.debug(f'Using single file mode with {str(tetrode_files)}')
    else:
        tetrode_files = sorted([tf.resolve() for tf in target_path.glob(cfg['TARGET_FILE_GLOB'])])
    logger.debug(f'Targets found: {tetrode_files}')

    from multiprocessing.pool import ThreadPool
    num_procs = cli_args.num_proc if cli_args.num_proc > 0 else len(tetrode_files)
    pool = ThreadPool(processes=num_procs)

    # for tfp in tetrode_files:
    params = [(cfg, tfp) for tfp in tetrode_files]
    params.append(cli_args.kkargs)
    print(params)

    results = pool.map_async(run_kk, params)

    pool.close()
    pool.join()

    print(results)


if __name__ == '__main__':
    main(sys.argv[1:])
