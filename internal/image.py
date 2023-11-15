from PIL import Image
import numpy as np


def extract_image(img):
    img = np.asarray(img * 255, dtype=np.uint8)
    img = np.squeeze(img)
    image = Image.fromarray(img)
    return image
