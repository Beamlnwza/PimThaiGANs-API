from dotenv import load_dotenv
import boto3
import os

load_dotenv(dotenv_path=".env")


def s3session():
    session = boto3.Session(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    return session


def s3client():
    session = s3session()
    s3_client = session.client("s3")
    return s3_client


def s3resource():
    session = s3session()
    s3_resource = session.resource("s3")
    return s3_resource
