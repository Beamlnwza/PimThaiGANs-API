from PIL import Image
import numpy as np
from io import BytesIO
from libs.s3 import s3_upload_image


def extract_image(img):
    img = np.asarray(img * 255, dtype=np.uint8)
    img = np.squeeze(img)
    image = Image.fromarray(img)
    return image


def image_byte(uuid, img_urls, index, img):
    img_bytesio = BytesIO()
    try:
        img.save(img_bytesio, format="JPEG")
        img_bytesio.seek(0)

        img_url = s3_upload_image(img_bytesio, f"{str(index).zfill(2)}.jpg", uuid)
        img_urls[str(index)] = img_url
    finally:
        img_bytesio.close()
