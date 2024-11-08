import io
import logging
import os
from typing import BinaryIO, Union

import boto3
import dotenv
from botocore.exceptions import ClientError

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# Get the S3 bucket name
BUCKET_AWS_NAME = os.getenv('BUCKET_AWS_NAME')


def put_object_to_s3(
    object_key: str,
    file_content: Union[str, bytes, BinaryIO],
    content_type: str = 'application/octet-stream',
    bucket_name: str = BUCKET_AWS_NAME,
) -> bool:
    """
    Save an in-memory file to an S3 bucket.

    This function can handle various file types including text, images, CSVs, and PDFs.

    Args:
        object_key: The key (path) where the object will be stored in the S3 bucket.
        file_content: The content of the file to be uploaded.
        content_type: The MIME type of the file. Defaults to 'application/octet-stream'.
        bucket_name: The name of the S3 bucket.
            Default to the environment variable `BUCKET_AWS_NAME`.

    Returns:
        True if the upload was successful, False otherwise.
    """
    # Connect to Amazon S3
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('BUCKET_AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('BUCKET_AWS_SECRET_ACCESS_KEY'),
    )

    try:
        if isinstance(file_content, str):
            file_content = file_content.encode('utf-8')
        elif isinstance(file_content, io.IOBase):
            file_content = file_content.read()

        # Upload the file to S3
        s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=file_content, ContentType=content_type)

        return True

    except ClientError as e:
        logger.warning(f'An error occurred: {e}')
        return False


def get_object_from_s3(
    object_key: str,
    bucket_name: str = BUCKET_AWS_NAME,
    stream: bool = False,
) -> Union[bytes, BinaryIO]:
    """
    Retrieve an object from an S3 bucket.

    This function can return the object content as bytes or as a file-like object.

    Args:
        object_key: The key (path) of the object in the S3 bucket.
        bucket_name: The name of the S3 bucket.
            Default to the environment variable `BUCKET_AWS_NAME`.
        stream: If True, returns a file-like object.Otherwise, returns the content as bytes.
            Defaults to False.
    Returns:
        The content of the S3 object as bytes or a file-like object.
    """
    # Connect to Amazon S3
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('BUCKET_AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('BUCKET_AWS_SECRET_ACCESS_KEY'),
    )

    try:
        if stream:
            # Get the object as a streaming body
            response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
            return response['Body']
        else:
            # Get the object content as bytes
            response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
            return response['Body'].read()

    except ClientError as e:
        logger.warning(f'An error occurred while retrieving the object: {e}')
        raise e


def upload_local_file_to_s3(
    object_key: str,
    file_path: str,
    bucket_name: str = BUCKET_AWS_NAME,
) -> bool:
    """
    Upload a local file to an S3 bucket.

    Args:
        object_key: The key (path) where the object will be stored in the S3 bucket.
        file_path: The local path of the file to be uploaded.
        bucket_name: The name of the S3 bucket.
            Default to the environment variable `BUCKET_AWS_NAME`.

    Returns:
        True if the upload was successful, False otherwise.
    """
    # Connect to Amazon S3
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('BUCKET_AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('BUCKET_AWS_SECRET_ACCESS_KEY'),
    )

    try:
        # Upload the file to S3
        s3_client.upload_file(file_path, bucket_name, object_key)
        return True

    except (ClientError, FileNotFoundError) as e:
        logger.warning(f'An error occurred while uploading the file: {e}')
        return False


def download_file_from_s3(
    object_key: str,
    file_path: str,
    bucket_name: str = BUCKET_AWS_NAME,
) -> bool:
    """
    Download a file from an S3 bucket to a local path.

    Args:
        object_key: The key (path) of the object in the S3 bucket.
        file_path: The local path where the file will be saved.
        bucket_name: The name of the S3 bucket.
            Default to the environment variable `BUCKET_AWS_NAME`.

    Returns:
        True if the download was successful, False otherwise.
    """
    # Connect to Amazon S3
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('BUCKET_AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('BUCKET_AWS_SECRET_ACCESS_KEY'),
    )

    try:
        # Download the file from S3
        s3_client.download_file(bucket_name, object_key, file_path)
        return True

    except (ClientError, FileNotFoundError) as e:
        logger.warning(f'An error occurred while downloading the file: {e}')
        return False
