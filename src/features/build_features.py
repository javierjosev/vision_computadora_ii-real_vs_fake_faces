# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from torchvision import transforms
from torch.utils.data import DataLoader

import src.common.persistence as persistence
import src.common.plots as plots


@click.command()
@click.argument('raw_path', type=click.Path(exists=True))
@click.argument('processed_path', type=click.Path())
def main(raw_path, processed_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final processed dataset from raw data')

    # Loading the raw dataloader
    logger.info('Loading the raw dataloader...')
    raw_train_dataset = persistence.load_dataset(raw_path, 'raw_train_dataset')
    raw_valid_dataset = persistence.load_dataset(raw_path, 'raw_valid_dataset')

    raw_train_dataset_aug = persistence.load_dataset(raw_path, 'raw_train_dataset')
    raw_valid_dataset_aug = persistence.load_dataset(raw_path, 'raw_valid_dataset')

    raw_train_dataset_aug_reduced = persistence.load_dataset(raw_path, 'raw_train_dataset')
    raw_valid_dataset_aug_reduced = persistence.load_dataset(raw_path, 'raw_valid_dataset')

    logger.info('Creating the transformations...')
    images_width = 224
    images_height = 224

    # mean_train & std_train were calculated in the notebooks
    mean_train = [0.521, 0.4259, 0.381]
    std_train = [0.248, 0.223, 0.221]

    transform = transforms.Compose([
        transforms.Resize((images_width, images_height)),
        transforms.ToTensor(),
        transforms.Normalize(mean_train, std_train),
    ])

    transform_aug = transforms.Compose([
        transforms.Resize((images_width, images_height)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(size=(images_width, images_height), scale=(0.8, 1.0)),
        transforms.ColorJitter(saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean_train, std_train)
    ])

    transform_aug_reduced = transforms.Compose([
        transforms.Resize((images_width, images_height)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomResizedCrop(size=(images_width, images_height), scale=(0.9, 1.0)),
        # transforms.ColorJitter(saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean_train, std_train)
    ])

    processed_train_dataloader, processed_valid_dataloader = build_data_loader("data", raw_train_dataset,
                                                                               raw_valid_dataset,
                                                                               transform, transform,
                                                                               32, processed_path, logger)

    processed_train_dataloader_aug, processed_valid_dataloader_aug = build_data_loader("data_aug",
                                                                                       raw_train_dataset_aug,
                                                                                       raw_valid_dataset_aug,
                                                                                       transform_aug, transform,
                                                                                       32, processed_path, logger)

    processed_train_dataloader_aug_reduced, processed_valid_dataloader_aug_reduced = build_data_loader(
                                                                                    "data_aug_reduced",
                                                                                    raw_train_dataset_aug_reduced,
                                                                                    raw_valid_dataset_aug_reduced,
                                                                                    transform_aug_reduced, transform,
                                                                                    32, processed_path, logger)

    plots.show_transformed_images(processed_train_dataloader.dataset, processed_train_dataloader_aug.dataset)
    plots.show_transformed_images(processed_train_dataloader.dataset, processed_train_dataloader_aug_reduced.dataset)


def build_data_loader(name, train_dataset, valid_dataset, train_transform, valid_transform, batch_size, processed_path,
                      logger):
    logger.info('Generating the dataloader')

    train_dataset.transform = train_transform
    valid_dataset.transform = valid_transform
    processed_train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    processed_valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    # Persisting the dataloaders
    persistence.save_dataloader(processed_train_dataloader, processed_path, 'processed_train_dataloader-' + name)
    persistence.save_dataloader(processed_valid_dataloader, processed_path, 'processed_valid_dataloader-' + name)

    return processed_train_dataloader, processed_valid_dataloader


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
