# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from torch.utils.data import DataLoader
import pandas as pd

from src.common.datasets import FacesDataset
import src.common.persistence as persistence


@click.command()
@click.argument('raw_path', type=click.Path(exists=True))
def main(raw_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        a dataset ready to be transformed (saved in ../raw).
    """
    logger = logging.getLogger(__name__)
    logger.info('Generating dataset from raw data')

    # Loading train & validation datasets
    train_df = pd.read_csv('data/raw/train.csv')
    valid_df = pd.read_csv('data/raw/valid.csv')

    # Generating datasets & data loader
    train_dataset = FacesDataset(dataframe=train_df, root_dir=raw_path)
    valid_dataset = FacesDataset(dataframe=valid_df, root_dir=raw_path)

    persistence.save_dataset(train_dataset, raw_path, 'raw_train_dataset')
    persistence.save_dataset(valid_dataset, raw_path, 'raw_valid_dataset')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
