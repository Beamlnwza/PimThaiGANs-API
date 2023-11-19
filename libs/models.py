from torch.autograd import Variable

import torch.nn as nn
import torch

import numpy as np

import os

IMAGE_SHAPE = (1, 32, 32)


class Generator(nn.Module):
    def __init__(self, n_classes):
        """
        Initializes a Generator module from Conditioanl GAN.

        Args:
            n_classes (int): The number of classes for label embedding.

        """
        super(Generator, self).__init__()

        latent_dim = 100

        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.image_shape = (1, 32, 32)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.image_shape))),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.image_shape)
        return img


MODELS_PATH = os.path.join(os.path.dirname(__file__), "models")

model_configs = [
    {"name": "0043", "classes": 44},
    {"name": "4461", "classes": 18},
    {"name": "6276", "classes": 15},
    {"name": "7787", "classes": 11},
]

models = []


def model_config(Generator, MODELS_PATH, models, config):
    """
    Loads a pre-trained model from a file and appends it to the models list.

    Args:
        Generator (torch.nn.Module): The generator model class.
        MODELS_PATH (str): The path to the directory containing the model files.
        models (list): The list of models to append the loaded model to.
        config (dict): The configuration dictionary containing the model information.

    Returns:
        None
    """
    model = Generator(n_classes=config["classes"])
    model_path = os.path.join(MODELS_PATH, f"{config['name']}.tar")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    models.append(model)


for config in model_configs:
    model_config(Generator, MODELS_PATH, models, config)


def generate_image_index(index: int):
    """
    Generate an image based on the given index.

    Args:
        index (int): The index used to generate the image.

    Returns:
        numpy.ndarray: The generated image as a NumPy array.
    """
    latent_dim = 100
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (1, latent_dim * 2))))

    model, range_index = get_model_and_range(index)

    model.eval()

    class_labels = range_index.index(index)
    labels = Variable(torch.LongTensor([class_labels]))

    random_z = z[:, :latent_dim]
    img = model(random_z + np.random.randint(-2, 2), labels)

    img_np = img.detach().numpy()
    return img_np


def get_model_and_range(index):
    """
    Returns the model and range index based on the given index.

    Parameters:
    index (int): The index value.

    Returns:
    tuple: A tuple containing the model and range index.
    """
    if index <= 43:
        model, range_index = models[0], range(0, 44)
    elif index <= 61:
        model, range_index = models[1], range(44, 62)
    elif index <= 76:
        model, range_index = models[2], range(62, 77)
    else:
        model, range_index = models[3], range(77, 88)

    return model, range_index
