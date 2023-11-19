from PIL import Image
import numpy as np
from io import BytesIO
from libs.s3 import s3_upload_image


def extract_image(img):
    """
    Extracts an image from a given numpy array.

    Args:
        img (numpy.ndarray): The input image as a numpy array.

    Returns:
        PIL.Image.Image: The extracted image.
    """
    img = np.asarray(img * 255, dtype=np.uint8)
    img = np.squeeze(img)
    image = Image.fromarray(img)
    return image


def image_byte(uuid, img_urls, index, img):
    """
    Convert an image to bytes and upload it to S3.

    Args:
        uuid (str): The UUID of the image.
        img_urls (dict): A dictionary to store the image URLs.
        index (int): The index of the image.
        img (PIL.Image.Image): The image to be converted.

    Returns:
        None
    """
    img_bytesio = BytesIO()
    try:
        img.save(img_bytesio, format="JPEG")
        img_bytesio.seek(0)

        img_url = s3_upload_image(img_bytesio, f"{str(index).zfill(2)}.jpg", uuid)
        img_urls[str(index)] = img_url
    finally:
        img_bytesio.close()
