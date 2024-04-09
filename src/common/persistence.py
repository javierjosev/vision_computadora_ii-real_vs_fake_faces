import os

import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader

from src.common.models import FacesSimpleCNN


def save_dataset(dataloader: Dataset, path, file_name):
    file_path = os.path.join(path, file_name + '.pth')
    torch.save(dataloader, file_path)


def save_dataloader(dataloader: DataLoader, path, file_name):
    file_path = os.path.join(path, file_name + '.pth')
    torch.save(dataloader, file_path)


def save_model(model: Module, path, file_name):
    file_path = os.path.join(path, file_name + '.pth')
    torch.save(model.state_dict(), file_path)


def load_dataset(path, file_name) -> Dataset:
    file_path = os.path.join(path, file_name + '.pth')
    return torch.load(file_path)


def load_dataloader(path, file_name) -> DataLoader:
    file_path = os.path.join(path, file_name + '.pth')
    return torch.load(file_path)


def load_faces_simple_cnn_model(path, file_name) -> Module:
    model = FacesSimpleCNN()
    file_path = os.path.join(path, file_name + '.pth')
    model.load_state_dict(torch.load(file_path))
    return model
