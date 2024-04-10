# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import torch
import torchmetrics
from dotenv import find_dotenv, load_dotenv

import src.common.persistence as persistence
import src.common.plots as plots


@click.command()
@click.argument('processed_path', type=click.Path(exists=True))
@click.argument('model_path', type=click.Path(exists=True))
def main(processed_path, model_path):
    """ Runs models predictions using processed datasets (../processed).
        The model binaries are loaded from ../models.
    """
    logger = logging.getLogger(__name__)
    logger.info('Generating models predictions...')

    logger.info('Loading processed dataloaders for predictions...')

    processed_valid_dataloader = persistence.load_dataloader(processed_path, 'processed_valid_dataloader-data')
    processed_valid_dataloader_aug = persistence.load_dataloader(processed_path, 'processed_valid_dataloader-data_aug')
    processed_valid_dataloader_aug_reduced = persistence.load_dataloader(processed_path, 'processed_valid_dataloader'
                                                                                         '-data_aug_reduced')

    logger.info('Loading trained models for predictions...')
    faces_simple_cnn_data_model = persistence.load_faces_simple_cnn_model(model_path, 'faces_simple_cnn_data')
    faces_simple_cnn_data_aug_model = persistence.load_faces_simple_cnn_model(model_path, 'faces_simple_cnn_data_aug')
    faces_simple_cnn_data_aug_reduced_model = \
        persistence.load_faces_simple_cnn_model(model_path, 'faces_simple_cnn_data_aug_reduced')
    faces_improved_cnn_data_aug_reduced_model = \
        persistence.load_faces_improved_cnn_model(model_path, 'faces_improved_cnn_data_aug_reduced')
    final_faces_cnn_data_aug_reduced_model = \
        persistence.load_final_faces_cnn_model(model_path, 'final_faces_cnn_data_aug_reduced')
    resnet18_binary_data_aug_reduced_model = \
        persistence.load_resnet18_binary_model(model_path, 'resnet18_binary_data_aug_reduced')

    metric = torchmetrics.classification.BinaryAccuracy()

    logger.info('Generating trained models predictions plots...')

    logger.info('Generating predictions for faces_simple_cnn_data_model...')
    model_predictions(faces_simple_cnn_data_model, processed_valid_dataloader, metric)

    logger.info('Generating predictions for faces_simple_cnn_data_aug_model...')
    model_predictions(faces_simple_cnn_data_aug_model, processed_valid_dataloader_aug, metric)

    logger.info('Generating predictions for faces_simple_cnn_data_aug_reduced_model...')
    model_predictions(faces_simple_cnn_data_aug_reduced_model, processed_valid_dataloader_aug_reduced, metric)

    logger.info('Generating predictions for faces_improved_cnn_data_aug_reduced_model...')
    model_predictions(faces_improved_cnn_data_aug_reduced_model, processed_valid_dataloader_aug_reduced, metric)

    logger.info('Generating predictions for final_faces_cnn_data_aug_reduced_model...')
    model_predictions(final_faces_cnn_data_aug_reduced_model, processed_valid_dataloader_aug_reduced, metric)

    logger.info('Generating predictions for resnet18_binary_data_aug_reduced_model...')
    model_predictions(resnet18_binary_data_aug_reduced_model, processed_valid_dataloader_aug_reduced, metric)


def model_predictions(model, valid_dataloader, metric):
    if torch.cuda.is_available():
        model.to("cuda")
        metric.to("cuda")
    plots.evaluate_model(model, valid_dataloader, metric)
    plots.plot_predictions(model, valid_dataloader)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
