import os
import unittest

import boto3
import pandas
import requests
from botocore.exceptions import ClientError
from fpdf import FPDF

from utils.prod.s3_utils import (
    BUCKET_AWS_NAME,
    download_file_from_s3,
    get_object_from_s3,
    put_object_to_s3,
    upload_local_file_to_s3,
)


class TestS3Functions(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the test environment.

        This method initializes the S3 client and sets up the test bucket.
        It will be run before each test method.
        """
        self.bucket_name = BUCKET_AWS_NAME

        # Connect to Amazon S3
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('BUCKET_AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('BUCKET_AWS_SECRET_ACCESS_KEY'),
        )

        # Ensure the test bucket exists
        self.s3.head_bucket(Bucket=self.bucket_name)

    def tearDown(self) -> None:
        """
        Clean up after each test.

        This method deletes all objects in the test bucket after each test.
        """
        response = self.s3.list_objects_v2(Bucket=self.bucket_name)
        if 'Contents' in response:
            for item in response['Contents']:
                if item['Key'].startswith('test_s3'):
                    delete_keys = {'Objects': [{'Key': item['Key']}]}
                    self.s3.delete_objects(Bucket=self.bucket_name, Delete=delete_keys)

    def test_put_object_to_s3(self) -> None:
        """
        Test `utils.prod.s3_utils.put_object_to_s3`.

        This test saves an in-memory text file to S3 and verifies its content and type.
        """
        object_key = os.path.join('test_s3', 'test_memory_file.txt')
        file_content = 'This is a test file content.'
        content_type = 'text/plain'

        # Test the function
        result = put_object_to_s3(object_key, file_content, content_type, self.bucket_name)

        # Assert the result
        self.assertTrue(result)

        # Verify the file was uploaded
        response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
        self.assertEqual(response['ContentType'], content_type)
        self.assertEqual(response['Body'].read().decode('utf-8'), file_content)

    def test_put_image_to_s3(self) -> None:
        """
        Test saving an image to S3 using `utils.prod.s3_utils.put_object_to_s3`

        This test downloads an image from a URL, saves it to S3, and verifies its content.
        """
        object_key = os.path.join('test_s3', 'logotype_sambanova_orange.png')
        image_url = 'https://sambanova.ai/hubfs/logotype_sambanova_orange.png'

        # Download the image
        response = requests.get(image_url)
        image_content = response.content

        # Save the image to S3
        result = put_object_to_s3(object_key, image_content, 'image/png', self.bucket_name)

        # Assert the result
        self.assertTrue(result)

        # Verify the image was uploaded
        response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
        self.assertEqual(response['ContentType'], 'image/png')
        self.assertEqual(response['Body'].read(), image_content)

    def test_put_pdf_to_s3(self) -> None:
        """
        Test saving a PDF to S3 using `utils.prod.s3_utils.put_object_to_s3`.

        This test creates a simple PDF, saves it to S3, and verifies its content.
        """
        object_key = os.path.join('test_s3', 'test_pdf.pdf')

        # Create a simple PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', size=15)
        pdf.cell(200, 10, txt='Welcome to the AI Starter Kit.', ln=True, align='C')
        pdf_content = pdf.output()

        # Save the PDF to S3
        result = put_object_to_s3(object_key, pdf_content, 'application/pdf', self.bucket_name)

        # Assert the result
        self.assertTrue(result)

        # Verify the PDF was uploaded
        response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
        self.assertEqual(response['ContentType'], 'application/pdf')
        self.assertEqual(response['Body'].read(), pdf_content)

    def test_put_csv_to_s3(self) -> None:
        """
        Test saving a CSV to S3 using `utils.prod.s3_utils.put_object_to_s3`.

        This test generates a CSV file, saves it to S3, and verifies its content.
        """
        object_key = os.path.join('test_s3', 'test_csv.csv')

        # Generate a simple CSV
        data = {
            'Country': ['France', 'Italy', 'UK'],
            'Capital': ['Paris', 'Rome', 'London'],
        }

        df = pandas.DataFrame(data)
        csv_content = df.to_csv(index=False).encode('utf-8')

        # Save the CSV to S3
        result = put_object_to_s3(object_key, csv_content, 'text/csv', self.bucket_name)

        # Assert the result
        self.assertTrue(result)

        # Verify the CSV was uploaded
        response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
        self.assertEqual(response['ContentType'], 'text/csv')
        self.assertEqual(response['Body'].read(), csv_content)

    def test_get_object_from_s3_as_bytes(self) -> None:
        """
        Test `utils.prod.s3_utils.get_object_from_s3`.

        This test uploads a file to S3, retrieves it, and verifies its content.
        """
        object_key = os.path.join('test_s3', 'test_get_bytes.txt')
        file_content = b'This is a test file content.'

        # Upload a file to S3 for testing
        self.s3.put_object(Bucket=self.bucket_name, Key=object_key, Body=file_content)

        # Get the object content as bytes (stream = True)
        retrieved_content = get_object_from_s3(object_key, self.bucket_name, stream=True)

        # Assert the content matches
        self.assertEqual(retrieved_content.read(), file_content)

        # Get the object content as a file-like object (stream = False)
        retrieved_content_body = get_object_from_s3(object_key, self.bucket_name, stream=False)

        # Assert the content matches
        self.assertEqual(retrieved_content_body, file_content)

        # Non existent object
        object_key = os.path.join('text_s3', 'non_existent_object.txt')
        with self.assertRaises(ClientError):
            get_object_from_s3(object_key, self.bucket_name)

    def test_upload_local_file_to_s3(self) -> None:
        """
        Test `utils.prod.s3_utils.upload_local_file_to_s3`.

        This test creates a local file, uploads it to S3, and verifies its content.
        """
        object_key = os.path.join('test_s3', 'test_upload.txt')
        file_content = 'This is a test file for upload.'
        file_path = 'utils/prod/tests/test_upload.txt'

        # Create a local file
        with open(file_path, 'w') as f:
            f.write(file_content)

        try:
            # Test the function
            result = upload_local_file_to_s3(object_key, file_path, self.bucket_name)

            # Assert the result
            self.assertTrue(result)

            # Verify the file was uploaded
            response = self.s3.get_object(Bucket=self.bucket_name, Key=object_key)
            self.assertEqual(response['Body'].read().decode('utf-8'), file_content)

        finally:
            # Clean up local file
            os.remove(file_path)

        # Non existent file path
        non_existent_file_path = os.path.join('test_s3', 'non_existent_object.txt')
        result = upload_local_file_to_s3(object_key, non_existent_file_path, self.bucket_name)
        self.assertFalse(result)

    def test_download_file_from_s3(self) -> None:
        """
        Test `utils.prod.s3_utils.download_file_from_s3`.

        This test uploads a file to S3, downloads it, and verifies its content.
        """
        object_key = os.path.join('test_s3', 'test_download.txt')
        file_content = 'This is a test file for download.'
        file_path = 'test_download.txt'

        # Upload a file to S3 for testing download
        self.s3.put_object(Bucket=self.bucket_name, Key=object_key, Body=file_content)

        try:
            # Test the function
            result = download_file_from_s3(object_key, file_path, self.bucket_name)

            # Assert the result
            self.assertTrue(result)

            # Verify the file was downloaded correctly
            with open(file_path, 'r') as f:
                downloaded_content = f.read()
            self.assertEqual(downloaded_content, file_content)

        finally:
            # Clean up local file
            if os.path.exists(file_path):
                os.remove(file_path)

        # Non existent file path
        non_existent_file_path = os.path.join('test_s3', 'non_existent_object.txt')
        result = download_file_from_s3(object_key, non_existent_file_path, self.bucket_name)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
