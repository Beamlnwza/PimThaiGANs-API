from torch.autograd import Variable

import torch.nn as nn
import torch

import numpy as np

import os

IMAGE_SHAPE = (1, 32, 32)


class Generator(nn.Module):
    def __init__(self, n_classes):
        super(Generator, self).__init__()

        latent_dim = 100

        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.IMAGE_SHAPE = IMAGE_SHAPE

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
            nn.Linear(1024, int(np.prod(self.IMAGE_SHAPE))),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.IMAGE_SHAPE)
        return img


MODELS_PATH = os.path.join(os.path.dirname(__file__), "models")
model_0043 = Generator(n_classes=44)
model_0043.load_state_dict(
    torch.load(os.path.join(MODELS_PATH, "0043.tar"), map_location="cpu")
)

model_4461 = Generator(n_classes=18)
model_6276 = Generator(n_classes=15)
model_7787 = Generator(n_classes=11)


def generate_image_index(index: int):
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (1, 100))))

    range_index = []

    if index < 43:
        model = model_0043
        range_index = range(0, 44)
    elif index < 61:
        model = model_4461
        range_index = range(44, 62)
    elif index < 76:
        model = model_6276
        range_index = range(62, 77)
    else:
        model = model_7787
        range_index = range(77, 88)

    model.eval()

    class_labels = range_index.index(index)
    labels = Variable(torch.LongTensor([class_labels]))

    img = model_0043(z, labels)
    img_np = img.detach().numpy()
    return img_np


if __name__ == "__main__":
    index = 0
    generated_image = generate_image_index(index)
    print(generated_image.shape)

    import matplotlib.pyplot as plt

    plt.imshow(generated_image[0][0], cmap="gray")
    plt.savefig("test.png")
