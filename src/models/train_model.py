# -*- coding: utf-8 -*-
import sys

import click
import logging
from pathlib import Path

import torch
import torchmetrics
from dotenv import find_dotenv, load_dotenv

import src.common.persistence as persistence
import src.common.models as models
import src.common.plots as plots


@click.command()
@click.argument('processed_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path())
@click.argument('model_to_train')
def main(processed_path, model_path, model_to_train):
    """ Runs model training using processed datasets (../processed).
        The model binaries are saved in ../models.
    """
    logger = logging.getLogger(__name__)
    logger.info('Training models...')

    logger.info('Loading processed dataloaders for training models...')

    processed_train_dataloader = persistence.load_dataloader(processed_path, 'processed_train_dataloader-data')
    processed_valid_dataloader = persistence.load_dataloader(processed_path, 'processed_valid_dataloader-data')

    processed_train_dataloader_aug = persistence.load_dataloader(processed_path, 'processed_train_dataloader-data_aug')
    processed_valid_dataloader_aug = persistence.load_dataloader(processed_path, 'processed_valid_dataloader-data_aug')

    processed_train_dataloader_aug_reduced = persistence.load_dataloader(processed_path, 'processed_train_dataloader'
                                                                                         '-data_aug_reduced')
    processed_valid_dataloader_aug_reduced = persistence.load_dataloader(processed_path, 'processed_valid_dataloader'
                                                                                         '-data_aug_reduced')

    match model_to_train:
        case "faces_simple_cnn-data":
            logger.info('ONLY faces_simple_cnn model with data SELECTED')
            train_faces_simple_cnn('faces_simple_cnn_data', model_path, processed_train_dataloader,
                                   processed_valid_dataloader, epochs=10, logger=logger, show_plots=True)
        case "faces_simple_cnn-data_aug":
            logger.info('ONLY faces_simple_cnn model with data_aug SELECTED')
            train_faces_simple_cnn('faces_simple_cnn_data_aug', model_path, processed_train_dataloader_aug,
                                   processed_valid_dataloader_aug, epochs=35, logger=logger, show_plots=True)
        case "faces_simple_cnn-data_aug_reduced":
            logger.info('ONLY faces_simple_cnn model with data_aug_reduced SELECTED')
            train_faces_simple_cnn('faces_simple_cnn_data_aug_reduced', model_path,
                                   processed_train_dataloader_aug_reduced,
                                   processed_valid_dataloader_aug_reduced, epochs=20, logger=logger, show_plots=True)
        case "all":
            logger.info('ALL models SELECTED')
            train_faces_simple_cnn('faces_simple_cnn_data', model_path, processed_train_dataloader,
                                   processed_valid_dataloader, epochs=10, logger=logger, show_plots=False)
            train_faces_simple_cnn('faces_simple_cnn_data_aug', model_path, processed_train_dataloader_aug,
                                   processed_valid_dataloader_aug, epochs=35, logger=logger, show_plots=False)
            train_faces_simple_cnn('faces_simple_cnn_data_aug_reduced', model_path,
                                   processed_train_dataloader_aug_reduced,
                                   processed_valid_dataloader_aug_reduced, epochs=20, logger=logger, show_plots=False)
        case _:
            logger.error("Invalid option")
            sys.exit()


def train_faces_simple_cnn(model_name, model_path, processed_train_dataloader, processed_valid_dataloader, epochs,
                           logger, show_plots=False):
    # # Simple model, simple data
    logger.info('Training FacesSimpleCNN model with dataset')
    simple_model = models.FacesSimpleCNN()
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.0001)
    loss = torch.nn.BCELoss()
    metric = torchmetrics.classification.BinaryAccuracy()
    data = {"train": processed_train_dataloader, "valid": processed_valid_dataloader}
    history = models.train(simple_model, optimizer, loss, metric, data, epochs)

    if show_plots:
        plots.plot_history(history)
        plots.evaluate_model(simple_model, processed_valid_dataloader, metric)

    persistence.save_model(simple_model, model_path, model_name)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
