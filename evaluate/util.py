
import pandas as pd
from mlproject.util import load_config_from_yaml
from argparse import ArgumentParser

def _setup_parser_get_args():
    """ set up a parser to ingest and return the path to metrics.csv """
    parser = ArgumentParser()
    parser.add_argument("metrics_csv", help="Provide a metrics.csv file to plot")
    parser.add_argument("--save_file", required=False, default=None, help="Provide a metrics.csv file to plot")
    args = parser.parse_args()
    return args.metrics_csv, args.save_file


def get_metrics_csv_from_yaml(config_yaml: str, logs_version: int=0):
    config_dict = load_config_from_yaml(config_yaml)
    log_dir = config_dict['log_dir']
    metrics_csv = ''.join(log_dir, 'lightning_logs/version_', str(logs_version), '/metrics.csv')
    return metrics_csv

def prepare_dataframes_from_csv(metrics_csv: str) -> tuple:
    df = pd.read_csv(metrics_csv)
    train_df = df.copy().loc[:, ~df.columns.str.contains('val')].dropna()
    val_df = df.copy().loc[:, ~df.columns.str.contains('train')].dropna()
    return train_df, val_df


def prepare_dataframes_from_yaml(config_yaml: str, logs_version: int=0) -> tuple:
    metrics_csv = get_metrics_csv_from_yaml(config_yaml, logs_version)
    return prepare_dataframes_from_csv(metrics_csv=metrics_csv)