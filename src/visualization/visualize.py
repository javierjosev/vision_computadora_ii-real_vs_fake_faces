# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

from torchview import draw_graph
from dotenv import find_dotenv, load_dotenv

from src.common.models import FacesSimpleCNN
from src.common.models import FacesImprovedCNN
from src.common.models import FinalFacesCNN
from src.common.models import ResNet18Binary


@click.command()
def main():
    """ Plots models architectures
    """
    logger = logging.getLogger(__name__)
    logger.info('Generating models visualizations...')

    images_height = 224
    images_width = 224
    model_graph('FacesSimpleCNN', FacesSimpleCNN(), images_width, images_height)
    model_graph('FacesImprovedCNN', FacesImprovedCNN(), images_width, images_height)
    model_graph('FinalFacesCNN', FinalFacesCNN(), images_width, images_height)
    model_graph('ResNet18Binary', ResNet18Binary(), images_width, images_height)


def model_graph(architecture_name, model_class, images_width, images_height):
    architecture = architecture_name
    model = model_class
    model_arch_graph = draw_graph(model, input_size=(1, 3, images_width, images_height),
                                  graph_dir='TB', roll=True, expand_nested=True,
                                  graph_name=f'self_{architecture}', save_graph=True,
                                  filename=f'self_{architecture}', directory='reports/figures/architectures')
    model_arch_graph.visual_graph


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
