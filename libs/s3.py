from dotenv import load_dotenv
import os
import boto3
from botocore.config import Config

load_dotenv(".env")

session = boto3.session.Session()


def s3_client():
    client = session.client(
        "s3",
        aws_access_key_id=os.environ.get("SPACES_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("SPACES_SECRET_KEY"),
        endpoint_url="https://sgp1.digitaloceanspaces.com",
        config=Config(s3={"addressing_style": "virtual"}),
        region_name="sgp1",
    )
    return client


def s3_resource():
    resource = session.resource(
        "s3",
        aws_access_key_id=os.environ.get("SPACES_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("SPACES_SECRET_KEY"),
        endpoint_url="https://sgp1.digitaloceanspaces.com",
        config=Config(s3={"addressing_style": "virtual"}),
        region_name="sgp1",
    )
    return resource


def s3_upload_image(filepath, filename, uuid, bucket_name="pimthaigans"):
    """
    Uploads an image file to an S3 bucket and returns the public URL of the uploaded image.

    Args:
        filepath (str): The path to the image file.
        filename (str): The name of the image file.
        uuid (str): The UUID associated with the image.
        bucket_name (str, optional): The name of the S3 bucket. Defaults to "pimthaigans".

    Returns:
        str: The public URL of the uploaded image.
    """
    client = s3_client()
    client.upload_fileobj(
        filepath,
        bucket_name,
        f"images/{uuid}/{filename}",
        ExtraArgs={"ACL": "public-read"},
    )

    return f"https://{bucket_name}.sgp1.digitaloceanspaces.com/images/{uuid}/{filename}"
